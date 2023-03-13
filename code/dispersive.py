from cmath import phase
import re
from weakref import ref
import lmfit
import numpy as np

from dqdring_code.fitting.utils import (
    estimate_f0,
    configure_parameters,
    plot_resonator_fit,
)
from cq_analysis.utils import fit_cable_delay

RED_PLANCK_INV = 1 / (6.582119569e-16)  # units of 2*pi*Hz/eV


class DispersiveReflectionModel(lmfit.Model):
    """Reflectometry signal model combined with a dispersive shift from a double-dot.

    Combines the input-output theory model of [De Jong et al, 2019,
    https://doi.org/10.1103/PhysRevApplied.11.044061] for the dispersive shift imparted
    on a resonator connected to the gate of a double-quantum-dot with the complex
    transmission model of [Malinowski et al, 2022,
    https://doi.org/10.1103/PhysRevApplied.18.024032] including asymmetries in the
    signal imparted by reflections from the cryogenic amplifier in the circuit.
    We use roughly the same notation as the originators of the model did for a
    transmission-style resonator circuit: [Petersson et al, 2012,
    https://doi.org/10.1038/nature11559].
    """

    @staticmethod
    def excitation_energy(tc, det):
        """Excitation energy gap of a two-level double-quantum dot."""
        return np.sqrt(det**2 + 4 * tc**2)

    @staticmethod
    def displacement(det, tc, wr, gc, gamma):
        """g*chi: The dispersive shift imparted on the resonator by the double-dot."""
        if tc == 0:
            return 0
        ex_energy = DispersiveReflectionModel.excitation_energy(tc, det)
        gchi = (
            4 * gc**2 * tc**2 / (ex_energy**2 * (wr + 0.5j * gamma - ex_energy))
        )
        return gchi

    @staticmethod
    def s11(tc, det, wr, gc, ki, ke, w0, gamma):
        """The reflection coefficient of the resonator-double-dot system."""
        kappa = ki + ke
        shift = DispersiveReflectionModel.displacement(det, tc, w0, gc, gamma)
        return 1 + 1j * ke / (wr - w0 - 0.5j * kappa + shift)

    @staticmethod
    def s21_with_amp(
        wr,
        s11_resonator,
        w0,
        refl_amp,
        g_coupl,
        fcav,
        phi_amp,
        phase_slope,
        phase_offset,
        amp_slope,
        amp_offset,
    ):
        """Convert reflection coefficient into transmission through measurement circuit.

        Adds the effect of losses through coupling coefficient 'g' of the directional
        coupler, as well as reflection and losses from the cryogenic amplifier, forming
        a cavity characterized by resonance frequency 'fcav', reflection from the
        amplifier 'a', and a phase offset imparted by reflections 'phi'.
        """
        numerator = (
            1j
            * g_coupl
            * refl_amp
            * np.sqrt(1 - g_coupl**2)
            * np.exp(-1j * wr / fcav + phi_amp * 1j)
            * s11_resonator
        )
        denominator = (
            1
            - np.sqrt(1 - refl_amp**2)
            * (1 - g_coupl**2)
            * np.exp(-1j * wr / fcav + phi_amp * 1j)
            * s11_resonator
        )
        env_prefactor = amp_offset * (1.0 + amp_slope * (wr - w0) / w0) + 0j
        env_prefactor *= np.exp(1j * (phase_offset + phase_slope * wr / (2 * np.pi)))
        return env_prefactor * numerator / denominator

    @staticmethod
    def func(
        axis_data,
        tc,
        gc,
        ki,
        ke,
        w0,
        gamma,
        alpha,
        volt_offset,
        refl_amp,
        g_coupl,
        fcav,
        phi_amp,
        phase_slope,
        phase_offset,
        amp_slope,
        amp_offset,
    ):
        """Complex transmission through resonator circuit connected to a double-dot."""
        # Extract voltage and frequency data from single independent variable axis_data
        detuning = alpha * (axis_data[0] - volt_offset) * RED_PLANCK_INV
        wr = 2 * np.pi * axis_data[1]
        s11 = DispersiveReflectionModel.s11(
            tc=np.absolute(tc),
            det=detuning,  # Convert from voltage to level detuning in 2*pi*Hz
            wr=wr,
            gc=gc,
            ki=np.absolute(ki),
            ke=np.absolute(ke),
            w0=np.absolute(w0),
            gamma=gamma,
        )
        s21 = DispersiveReflectionModel.s21_with_amp(
            wr=wr,
            s11_resonator=s11,
            w0=w0,
            refl_amp=refl_amp,
            g_coupl=g_coupl,
            fcav=fcav,
            phi_amp=phi_amp,
            phase_slope=phase_slope,
            phase_offset=phase_offset,
            amp_slope=amp_slope,
            amp_offset=amp_offset,
        )
        return s21

    def __init__(self, **kwargs):
        """Constructor for the DispersiveReflectionModel class.

        Keyword Arguments:
        ------------------
        **kwargs (typing.Any): Keyword arguments to pass to the constructor for
            lmfit.Model.

        Returns:
        --------
        DispersiveReflectionModel: An instance of this class.
        """
        super().__init__(self.func, **kwargs)
        # Record known amplifier/coupler parameters for the measurement setup used.
        self._g_coupl_spec = 0.178
        self._refl_amp_spec = 0.398

    def guess(
        self,
        cdata,
        voltages,
        freqs,
        alpha_guess=0.3,
        fixed_phase_slope=None,
        verbose=False,
    ):
        """Create lmfit.Parameters with informed guesses based on input data."""
        p0 = self.make_params()

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
            )
        else:
            phase_slope = fixed_phase_slope
        p0["phase_slope"].set(
            value=phase_slope,
            min=-2e6,
            max=2e-6,  # bounds are based on what is realistically possible
            vary=slope_varies,
        )
        unwound_cdata = np.exp(-1j * phase_slope * freqs) * cdata

        tc = 20e-6  # in eV
        p0["tc"].set(value=tc * RED_PLANCK_INV, min=5e8, max=1000e9)
        gc = 100e6
        p0["gc"].set(value=gc, min=0)
        ki = 1e6
        p0["ki"].set(value=ki, min=0)
        ke = 3e6
        p0["ke"].set(value=ke, min=0)
        res_freqs = np.array(
            [
                2
                * np.pi
                * estimate_f0(freq_trace, freqs, return_index=False, verbose=False)
                for freq_trace in unwound_cdata
            ]
        )
        w0 = np.median(res_freqs)
        p0["w0"].set(
            value=w0, min=2 * np.pi * np.min(freqs), max=2 * np.pi * np.max(freqs)
        )
        gamma = 2e9
        p0["gamma"].set(value=gamma, min=0)
        p0["alpha"].set(value=alpha_guess, min=0.0, max=1)
        coulomb_res_index = np.argmin(res_freqs)
        volt_offset = voltages[coulomb_res_index]
        p0["volt_offset"].set(
            value=volt_offset, min=np.min(voltages), max=np.max(voltages) + 1e-9
        )
        p0["refl_amp"].set(value=self._refl_amp_spec, min=0, max=1)
        p0["g_coupl"].set(value=self._g_coupl_spec, min=0, max=1)
        p0["fcav"].set(value=220e6, min=1e7, max=1e10)
        p0["amp_offset"].set(value=1, min=0)
        p0["amp_slope"].set(value=0)
        p0["phase_offset"].set(value=0)
        return p0

    def quick_fit_no_dots(
        self,
        cdata,
        freqs,
        p0=None,
        overridden_guesses=dict(),
        fixed_params=dict(),
        fix_cavity_params=True,
        verbose=False,
        **kwargs,
    ):
        voltages = np.array([0.0])
        if p0 is None:
            if "phase_slope" in fixed_params.keys():
                fixed_phase_slope = fixed_params["phase_slope"]
            else:
                fixed_phase_slope = None
            p0 = self.guess(
                cdata,
                voltages=voltages,
                freqs=freqs,
                fixed_phase_slope=fixed_phase_slope,
                verbose=verbose,
            )
        if fix_cavity_params:
            p0["g_coupl"].vary = False
            p0["refl_amp"].vary = False
        # Ensure dot-related parameters are fixed such that only the resonator is
        # considered in the absence of the dots.
        fixed_params.update(
            {"gc": 0, "tc": 0, "volt_offset": 0, "gamma": 0, "alpha": 0}
        )
        p0 = configure_parameters(
            p0, overridden_guesses=overridden_guesses, fixed_params=fixed_params
        )
        # Use dummy value for voltage axis since it will have no effect in this fit
        axis_data = np.array(
            [
                np.array([0.0]),
                freqs,
            ]
        )
        fit_result = self.fit(cdata, params=p0, axis_data=axis_data, **kwargs)
        if verbose:
            # TODO: Check if the plot_resonator_function actually works in this way
            plot_resonator_fit(
                fit_result, freqs=freqs, res_freq_name="w0", angular_res_freq=True
            )
        return fit_result

    def quick_fit(
        self,
        cdata,
        freqs,
        voltages,
        p0=None,
        overridden_guesses=dict(),
        fixed_params=dict(),
        fix_cavity_params=True,
        verbose=False,
        **kwargs,
    ):
        if p0 is None:
            if "phase_slope" in fixed_params.keys():
                fixed_phase_slope = fixed_params["phase_slope"]
            else:
                fixed_phase_slope = None
            p0 = self.guess(
                cdata,
                voltages=voltages,
                freqs=freqs,
                fixed_phase_slope=fixed_phase_slope,
                verbose=verbose,
            )
        if fix_cavity_params:
            p0["g_coupl"].vary = False
            p0["refl_amp"].vary = False
        p0 = configure_parameters(
            p0, overridden_guesses=overridden_guesses, fixed_params=fixed_params
        )
        # Construct single independent variable out of frequencies and voltages
        axis_data = np.array([voltages[:, None], freqs[None, :]])
        fit_result = self.fit(cdata, params=p0, axis_data=axis_data, **kwargs)
        if verbose:
            self.plot_fit(fit_result)
        return fit_result

    def plot_fit(self, fit_result):
        # TODO: Implement this function
        print(f"NOT IMPLEMENTED YET. fit_result:\n {fit_result}")
