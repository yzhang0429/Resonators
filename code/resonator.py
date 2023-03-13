import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from .utils import algebraic_circle_fit, fit_cable_delay, estimate_f0


class ReflectionResonatorBase(Model):
    def guess(self):
        raise Exception(
            "ReflectionResonatorBase class must be inherited by another "
            "class also inheriting from lmfit.Model which defines its own "
            "guess function!"
        )

    @staticmethod
    def s11_resonator(freqs, f0, Qi_inv, Qe_inv, phi=0):
        Q = 1 / (Qi_inv + np.cos(phi) * Qe_inv)
        numerator = 2 * Q * Qe_inv * np.exp(1j * phi)
        denominator = 1 + 2 * 1j * Q * (freqs - f0) / f0
        return 1 - numerator / denominator

    @staticmethod
    def realistic_s21(
        freqs, f0, s21s, amp_slope, amp_offset, phase_winding, phase_offset
    ):
        sig = s21s * amp_offset * (1.0 + amp_slope * (freqs - f0) / f0)
        sig *= np.exp(1j * (phase_offset + phase_winding * freqs))
        return sig

    def quick_fit(
        self,
        cdata,
        freqs,
        p0=None,
        fixed_params={},
        overridden_guesses={},
        verbose=True,
        max_nfev=None,
        **kwargs,
    ):
        """Convenience function for fitting with optional parameter overrides.

        Parameters:
        -----------
        cdata (numpy.ndarray): Array of complex S21 data, assumed to
            have linear amplitude units
        freqs (numpy.ndarray): Measured frequencies corresponding to each
            cdata point.

        Keyword Arguments:
        ------------------
        p0 (lmfit.Parameters): Optionally input one's own Parameters as
            guesses and constraints for the fit. Defaults to generating p0
            from this class' own guess() function.
        fixed_params (dict): Dictionary of values to optionally fix certain
            parameters during the fit.
        overridden_guesses (dict): Dictionary of values to override this
            model's own guess function guess for specified parameters, unlike
            fixed_params, does not fix the parameters. If common parameters are
            in both of these keyword arguments, fixed_params takes precedence.
        max_nfev (optional int): Maximum number of function evaluations during the
            fitting process. Passed to lmfit.Model.fit()
        **kwargs: Keyword arguments to pass to the lmfit.Model's guess function.

        Returns:
        --------
        fit_result (lmfit.ModelResult): Result of the fit.
        """
        # Initialize guessed parameters if necessary
        if p0 is None:
            p0 = self.guess(
                cdata,
                freqs,
                fixed_phase_slope=(
                    fixed_params["phase_winding"]
                    if "phase_winding" in fixed_params
                    else None
                ),
                verbose=verbose,
                **kwargs,
            )
        # Optionally override guessed Parameter initial values
        for param_key, guess_val in overridden_guesses.items():
            p0[param_key].value = guess_val
        # Optionally fix Parameter values
        for param_key, fixed_val in fixed_params.items():
            p0[param_key].set(value=fixed_val, vary=False)
        fit_result = self.fit(cdata, params=p0, max_nfev=max_nfev, freqs=freqs)
        if verbose:
            print(fit_result.fit_report())
            fit_result.plot_fit()
        return fit_result


class ReflectionModelMalinowski(ReflectionResonatorBase):
    """
    The resonator model described in https://arxiv.org/abs/2110.03257 formulated in
    terms of inverse quality factors, and with modified guesses and guess bounds.

    Parameters:
    - f0 [Hz] - resonator frequency
    - Qi_inv - internal quality factor
    - Qe_inv - external quality factor
    - a [1] - reflection coefficient of the amplifier
            (0.398 for CITLF2 HEMT amp)
    - g [1] - coupling coefficient of the directional coupler
            (0.178 for MiniCrcuits ZEDC-15-2B)
    - fcav [Hz] - period of the background oscillations
            due to cavity between amp and the device
    - phi [rad] - phase offset at the amp reflection
            (should be 0, but may need to be left free in the final fit)
    - amp_slope [1] - additional slope of the transmission amplitude
    - amp_offset [measurement device units] - amplitude
        corresponding to the full transmission
    - phase_winding [rad/Hz]
    - phase_offset [rad]

    To ignore the reflection from the amp fix g=0 and a=1.
    """

    @staticmethod
    def s11_with_amp(freqs, s11, a, g, fcav, phi):
        numerator = (
            1j
            * g
            * a
            * np.sqrt(1 - g**2)
            * np.exp(-2j * np.pi * freqs / fcav + phi * 1j)
            * s11
        )
        denominator = (
            1
            - np.sqrt(1 - a**2)
            * (1 - g**2)
            * np.exp(-2j * np.pi * freqs / fcav + phi * 1j)
            * s11
        )
        return numerator / denominator

    @staticmethod
    def func(
        freqs,
        f0,
        Qi_inv,
        Qe_inv,
        a,
        g,
        fcav,
        phi,
        amp_slope,
        amp_offset,
        phase_winding,
        phase_offset,
    ):
        s11 = ReflectionResonatorBase.s11_resonator(freqs, f0, Qi_inv, Qe_inv)
        sig = ReflectionModelMalinowski.s11_with_amp(freqs, s11, a, g, fcav, phi)
        sig = ReflectionResonatorBase.realistic_s21(
            freqs, f0, sig, amp_slope, amp_offset, phase_winding, phase_offset
        )
        return sig

    def __init__(self, **kwargs):
        super().__init__(self.func, **kwargs)
        self._g_spec = 0.178
        self._a_spec = 0.398

    def guess(
        self,
        cdata,
        freqs,
        fixed_phase_slope=None,
        edge_fracs=(0.05, 0.05),
        fix_cavity_params=True,
        verbose=False,
    ):
        p0 = self.make_params()  # Initialize Parameters
        # Approximate S21 data far off resonance as average of number of points near edge of freq. window
        # determined by edge_fracs
        edge_dists = (int(freqs.size * edge_fracs[0]), int(freqs.size * edge_fracs[1]))
        edge_left, edge_right = np.mean(cdata[: edge_dists[0]]), np.mean(
            cdata[: -edge_dists[1]]
        )
        freq_left, freq_right = np.mean(freqs[: edge_dists[0]]), np.mean(
            freqs[: -edge_dists[1]]
        )
        # Define bounds for periodic parameters, trying to prevent lmfit from picking
        # boundary values by including more than 2*pi, without making interval infinite
        phase_bounds = (-np.pi - 0.1, +np.pi + 0.1)
        abs_cdata = np.abs(cdata)

        # Calibrate phase winding if not already input
        slope_varies = (fixed_phase_slope is None) or (fixed_phase_slope is False)
        if fixed_phase_slope is None or (
            type(fixed_phase_slope) == bool and fixed_phase_slope is True
        ):
            phase_slope = fit_cable_delay(
                freqs,
                cdata,
                verbose=verbose,
                return_full_result=False,
                fit_offset=False,
            )
        else:
            phase_slope = fixed_phase_slope
        p0["phase_winding"].set(
            value=phase_slope,
            min=-2e6,
            max=2e-6,  # bounds are based on what is realistically possible
            vary=slope_varies,
        )
        unwound_cdata = np.exp(-1j * phase_slope * freqs) * cdata

        # Guess f0 based on region where slope of phase is maximal after removing phase roll
        res_index = estimate_f0(
            unwound_cdata, freqs, return_index=True, verbose=verbose
        )
        f0 = freqs[res_index]
        freq_width = np.max(freqs) - np.min(freqs)
        p0["f0"].set(
            value=f0,
            min=max([0, np.min(freqs) - freq_width]),
            max=np.max(freqs) + freq_width,
            vary=True,
        )

        # Naive guesses for ki and ke assuming Qi and Qe are small and that resonator is overcoupled
        abs_std = np.std(abs_cdata)
        shoulder_freq = freqs[
            np.argmin(np.abs(abs_cdata - abs_cdata[res_index] + abs_std))
        ]
        peak_width = 2 * np.abs(shoulder_freq - f0)
        Qi_inv = peak_width / f0
        # Conduct algebraic circle fit of unwound data to estimate Qe in the scenario where there is
        # no asymmetry in the amplifier, as a first approximation
        circle_fit = algebraic_circle_fit(unwound_cdata)
        center = (
            -0.5 * (circle_fit[1] + 1j * circle_fit[2]) / circle_fit[0]
        )  # Not needed here, just FYI
        diameter = 1 / np.abs(circle_fit[0][0])
        if verbose:
            plt.figure()
            plt.plot(unwound_cdata.real, unwound_cdata.imag, "k.")
            angles = np.linspace(0, 2 * np.pi)
            plt.plot(
                center.real + 0.5 * diameter * np.cos(angles),
                center.imag + 0.5 * diameter * np.sin(angles),
                "b-",
            )
            plt.show()
        Qe_inv = (
            Qi_inv * diameter / (1 - diameter)
        )  # Calculate Qe assuming resonator is circle of diameter Q/Qe
        p0["Qi_inv"].set(value=Qi_inv, min=1 / 10_000, max=10)
        p0["Qe_inv"].set(value=Qe_inv, min=1 / 1_000, max=10)

        # Assume no phase offset from cavity initially
        phi = 0
        p0["phi"].set(value=phi, min=phase_bounds[0], max=phase_bounds[1])

        # Guess fcav based on what is seen typically in B1. Oscillations of background occur
        # on the scale of our band, which is about 100-1000MHz, so fcav should have a periodicity
        # less than or not too far above that bandwidth
        fcav = 220e6
        p0["fcav"].set(
            value=fcav,
            min=10e6,
            #                        max=2*1000e6
        )

        off_res_s21 = 0.5 * (edge_left + edge_right)
        # Guess phase offset as phase which cancels out full S21 phase including
        # amp when resonator is far off resonance (such that the bare res. S11=1)
        phase_offset = -np.angle(off_res_s21)
        p0["phase_offset"].set(
            value=phase_offset, min=phase_bounds[0], max=phase_bounds[1]
        )

        # Guess amp_offset and slope again assuming the amplifier cavity has no influence on the
        # resonance as a lowest order approximation
        # amp offset should be the value of the straight line connecting the two edge points evaluated
        # at the resonance frequency
        background_slope = (np.abs(edge_right) - np.abs(edge_left)) / (
            freq_right - freq_left
        )
        zero_intercept = np.abs(edge_left) - background_slope * freq_left
        amp_offset = background_slope * f0 + zero_intercept
        amp_slope = background_slope * f0
        p0["amp_slope"].set(value=amp_slope)
        p0["amp_offset"].set(
            value=amp_offset,
            min=0.0001 * np.min(abs_cdata),
            #                              max=100*np.max(abs_cdata)
        )

        a = self._a_spec
        g = self._g_spec
        p0["a"].set(value=a, min=0.001, max=1, vary=not fix_cavity_params)
        p0["g"].set(value=g, min=0.001, max=1, vary=not fix_cavity_params)
        return p0


class ReflectionModelKhalil(ReflectionResonatorBase):
    """
    The resonator model described in https://arxiv.org/abs/1410.3365 formulated in
    terms of inverse quality factors, adapted to a reflection (instead of notch-style)
    resonator and with modified guesses and guess bounds.

    Parameters:
    - f0 [Hz] - resonator frequency
    - Qi_inv - internal quality factor
    - Qe_inv - external quality factor
    - phi [rad] - Complex phase of Q_e, relating to impedance mismatches in resonator
        circuit granting the resonator lineshape asymmetry
    - amp_slope [1] - additional slope of the transmission amplitude
    - amp_offset [measurement device units] - amplitude
        corresponding to the full transmission
    - phase_winding [rad/Hz]
    - phase_offset [rad]
    """

    @staticmethod
    def func(
        freqs,
        f0,
        Qi_inv,
        Qe_inv,
        phi,
        amp_slope,
        amp_offset,
        phase_winding,
        phase_offset,
    ):
        s11 = ReflectionResonatorBase.s11_resonator(freqs, f0, Qi_inv, Qe_inv, phi=phi)
        sig = ReflectionResonatorBase.realistic_s21(
            freqs, f0, s11, amp_slope, amp_offset, phase_winding, phase_offset
        )
        return sig

    def __init__(self, **kwargs):
        super().__init__(self.func, **kwargs)
        self._g_spec = 0.178
        self._a_spec = 0.398

    def guess(
        self,
        cdata,
        freqs,
        fixed_phase_slope=None,
        edge_fracs=(0.05, 0.05),
        verbose=False,
    ):
        p0 = self.make_params()  # Initialize Parameters
        # Approximate S21 data far off resonance as average of number of points near edge of freq. window
        # determined by edge_fracs
        edge_dists = (int(freqs.size * edge_fracs[0]), int(freqs.size * edge_fracs[1]))
        edge_left, edge_right = np.mean(cdata[: edge_dists[0]]), np.mean(
            cdata[: -edge_dists[1]]
        )
        freq_left, freq_right = np.mean(freqs[: edge_dists[0]]), np.mean(
            freqs[: -edge_dists[1]]
        )
        # Define bounds for periodic parameters, trying to prevent lmfit from picking
        # boundary values by including more than 2*pi, without making interval infinite
        phase_bounds = (-np.pi - 0.1, +np.pi + 0.1)
        abs_cdata = np.abs(cdata)

        # Calibrate phase winding if not already input
        slope_varies = (fixed_phase_slope is None) or (fixed_phase_slope is False)
        if fixed_phase_slope is None or (
            type(fixed_phase_slope) == bool and fixed_phase_slope is True
        ):
            phase_slope = fit_cable_delay(
                freqs,
                cdata,
                verbose=verbose,
                return_full_result=False,
                fit_offset=False,
            )
        else:
            phase_slope = fixed_phase_slope
        p0["phase_winding"].set(
            value=phase_slope,
            min=-2e6,
            max=2e-6,  # bounds are based on what is realistically possible
            vary=slope_varies,
        )
        unwound_cdata = np.exp(-1j * phase_slope * freqs) * cdata

        # Guess f0 based on region where slope of phase is maximal after removing phase roll
        res_index = estimate_f0(
            unwound_cdata, freqs, return_index=True, verbose=verbose
        )
        f0 = freqs[res_index]
        freq_width = np.max(freqs) - np.min(freqs)
        p0["f0"].set(
            value=f0,
            min=max([0, np.min(freqs) - freq_width]),
            max=np.max(freqs) + freq_width,
            vary=True,
        )

        # Naive guesses for ki and ke assuming Qi and Qe are small and that resonator is overcoupled
        abs_std = np.std(abs_cdata)
        shoulder_freq = freqs[
            np.argmin(np.abs(abs_cdata - abs_cdata[res_index] + abs_std))
        ]
        peak_width = 2 * np.abs(shoulder_freq - f0)
        Qi_inv = peak_width / f0
        # Conduct algebraic circle fit of unwound data to estimate Qe in the scenario where there is
        # no asymmetry in the amplifier, as a first approximation
        circle_fit = algebraic_circle_fit(unwound_cdata)
        center = (
            -0.5 * (circle_fit[1] + 1j * circle_fit[2]) / circle_fit[0]
        )  # Not needed here, just FYI
        diameter = 1 / np.abs(circle_fit[0][0])
        if verbose:
            plt.figure()
            plt.plot(unwound_cdata.real, unwound_cdata.imag, "k.")
            angles = np.linspace(0, 2 * np.pi)
            plt.plot(
                center.real + 0.5 * diameter * np.cos(angles),
                center.imag + 0.5 * diameter * np.sin(angles),
                "b-",
            )
            plt.show()
        Qe_inv = (
            Qi_inv * diameter / (1 - diameter)
        )  # Calculate Qe assuming resonator is circle of diameter Q/Qe
        p0["Qi_inv"].set(value=Qi_inv, min=1 / 1_000_000, max=10)
        p0["Qe_inv"].set(value=Qe_inv, min=1 / 1_000, max=10)

        # Assume no resonator asymmetry by default
        phi = 0
        p0["phi"].set(value=phi, min=phase_bounds[0], max=phase_bounds[1])

        off_res_s21 = 0.5 * (edge_left + edge_right)
        # Guess phase offset as phase which cancels out full S21 phase including
        # amp when resonator is far off resonance (such that the bare res. S11=1)
        phase_offset = -np.angle(off_res_s21)
        p0["phase_offset"].set(
            value=phase_offset, min=phase_bounds[0], max=phase_bounds[1]
        )

        # Guess amp_offset and slope again assuming the amplifier cavity has no influence on the
        # resonance as a lowest order approximation
        # amp offset should be the value of the straight line connecting the two edge points evaluated
        # at the resonance frequency
        background_slope = (np.abs(edge_right) - np.abs(edge_left)) / (
            freq_right - freq_left
        )
        zero_intercept = np.abs(edge_left) - background_slope * freq_left
        amp_offset = background_slope * f0 + zero_intercept
        amp_slope = background_slope * f0
        p0["amp_slope"].set(value=amp_slope)
        p0["amp_offset"].set(
            value=amp_offset,
            min=0.0001 * np.min(abs_cdata),
            #                              max=100*np.max(abs_cdata)
        )
        return p0
