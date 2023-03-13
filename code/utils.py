import numpy as np
import lmfit
from scipy.optimize import newton
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import svd
from matplotlib.pyplot import subplots, figure, setp, show
from matplotlib.gridspec import GridSpec


def configure_parameters(params, overridden_guesses=dict(), fixed_params=dict()):
    """Fit an lmfit.Model while overriding and fixing some parameters.

    Parameters:
    -----------
    params (lmfit.Parameters): Parameters object of an lmfit.Model to be modified.

    Keyword Arguments:
    ------------------
    overridden_guesses (dict, optional): Dictionary of parameter values indexed by the
        lmfit.Parameter label to change the params value to for that parameter without
        changing whether or not that parameter can vary or not.
    fixed_params (dict, optional): Dictionary of parameter values indexed by the
        lmfit.Parameter label to change the params value to while also fixing that
        parameter.

    Returns:
    --------
    lmfit.Parameters: Modified parameters with some values changed and some variables
        fixed.
    """
    for param_key, guess_val in overridden_guesses.items():
        params[param_key].value = guess_val
    for param_key, fixed_val in fixed_params.items():
        params[param_key].set(value=fixed_val, vary=False)
    return params


def estimate_f0(
    cdata: np.ndarray,
    freqs: np.ndarray,
    phase_winding: float = 0,
    return_index: bool = False,
    verbose: bool = False,
) -> float:
    """Guesses f0 as point of maximum phase slope after unwinding/smoothing.

    Parameters:
    -----------
    cdata (numpy.ndarray): One-dimensional array of complex resonator transmission
        data to estimate the resonance frequency of.
    freqs (numpy.ndarray): Measured frequencies corresponding to each point in cdata.

    Keyword Arguments:
    ------------------
    phase_winding (float = 0): Slope offset of phase of complex data due to the finite
        time required for light to propagate through the whole transmission circuit.
        In units of [rad/Hz].
    return_index (bool = False): Whether or not to return the index in freqs of the
        resonance frequency estimate, or to simply return the resonance frequency
        estimate itself.
    verbose (bool = False): Whether or not to plot cdata in the complex plane, along
        with the estimated resonance frequency.

    Returns:
    --------
    float: Either the index of the estimated resonance frequency (if return_index is
        True) or the estimated frequency itself.
    """
    unwound_data = cdata * np.exp(-1j * phase_winding * freqs)
    unwound_phases = np.angle(unwound_data)
    unwound_phases = gaussian_filter1d(unwound_phases, sigma=1)
    f0_index = np.argmax(np.abs(np.diff(unwound_phases)))
    f0 = freqs[f0_index]
    if verbose:
        # Plot data in complex plane after subtracting phase roll
        _, axs = subplots(1, 2, figsize=(10, 5))
        axs[0].plot(freqs / 1e6, unwound_phases)
        axs[0].axvline(f0 / 1e6, color="green", alpha=0.5)
        axs[0].set_xlabel("Frequency (MHz)")
        axs[0].set_ylabel(f"Arg[S21]*exp(-i{phase_winding}*freqs) (rad)")
        axs[1].plot(unwound_data.real, unwound_data.imag, "k-")
        axs[1].plot(unwound_data.real[f0_index], unwound_data.imag[f0_index], "gx")
        axs[1].set_xlabel(f"Re[S21*exp(-i{phase_winding}*freqs)]")
        axs[1].set_ylabel(f"Im[S21*exp(-i{phase_winding}*freqs)]")
    return f0_index if return_index else f0


def log_to_linear(log_data):
    """Convert from base-10 logarithmic data to linear data."""
    return 10 ** (log_data / 10)


def environment_prefactor(
    freqs,
    params,
    res_freq_kw="f0",
    phase_slope_kw="phase_winding",
    phase_offset_kw="phase_offset",
    amp_offset_kw="amp_offset",
    amp_slope_kw="amp_slope",
):
    phase_factor = np.exp(
        1j * (params[phase_offset_kw] + freqs * params[phase_slope_kw])
    )
    amp_factor = params[amp_offset_kw] * (
        1 + params[amp_slope_kw] * (freqs - params[res_freq_kw]) / params[res_freq_kw]
    )
    return phase_factor * amp_factor


def plot_resonator_fit(
    fit_result,
    freqs=None,
    res_freq_name="f0",
    normalize_residuals=False,
    angular_res_freq=False,
    unwind_phase=True,
    divide_out_environment=True,
    plot_origin=False,
):
    """Create detailed plot of a frequency-dependent fit of complex resonator data.

    Parameters:
    -----------
    fit_result (lmfit.ModelResult): Fit result of an lmfit.Model for a complex-valued
        function of a single variable modelling a resonator response.

    Keyword Arguments:
    ------------------
    freqs ([float,], optional): Array of frequencies to override independent variable
        data in fit_result optionally.
    res_freq_name (str = "f0"): String keyword to access the resonance frequency of
        the resonator model.
    normalize_residuals (bool = False): Whether or not to normalize the plotted
        fit residuals by the mean of the complex data's magnitude.
    angular_res_freq (bool = False): Whether or not the resonance frequency in the
        fit result is angular or not.
    unwind_phase (bool = True): Whether or not to divide out the phase_winding slope
        extracted from the best fit from all of the plots.
    divide_out_environment (bool = True): Whether or not to divide out the phase slope,
        phase offset, amplitude slope, and amplitude offset of the data and fit
        before plotting. If true, overrides unwind_phase even if unwind_phase is False.
    plot_origin (bool=False): Whether or not to plot horizontal and vertical rules
        showing the origin and I=1.
    """
    fr = fit_result
    # Get frequency, assuming it is the first independent variable of the lmfit.Model
    if freqs is None:
        freqs = fr.userkws[fr.model.independent_vars[0]]
    mag_residuals = np.abs(fr.data - fr.best_fit)
    if normalize_residuals:
        # Convert residuals to a percentage of the data's mean value
        mag_residuals *= 100 / np.mean(np.abs(fr.data))
    res_index = np.argmin(
        np.abs(
            freqs
            - fr.best_values[res_freq_name] / (2 * np.pi if angular_res_freq else 1)
        )
    )
    data, init_fit, best_fit = fr.data, fr.init_fit, fr.best_fit
    if divide_out_environment:
        best_env_prefactor = environment_prefactor(freqs, fr.best_values)
        init_env_prefactor = environment_prefactor(freqs, fr.init_values)
        data /= best_env_prefactor
        best_fit /= best_env_prefactor
        init_fit /= init_env_prefactor
    elif unwind_phase:
        best_phase_prefactor = np.exp(1j * freqs * fr.best_values["phase_winding"])
        init_phase_prefactor = np.exp(1j * freqs * fr.init_values["phase_winding"])
        data /= best_phase_prefactor
        best_fit /= best_phase_prefactor
        init_fit /= init_phase_prefactor

    fig = figure(figsize=(12.5, 6))
    gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 2])
    ax0 = fig.add_subplot(gs[:, 0])
    ax0.plot(data.real, data.imag, "k.")
    ax0.plot(init_fit.real, init_fit.imag, "r-")
    ax0.plot(best_fit.real, best_fit.imag, "g-")
    ax0.plot(best_fit.real[res_index], best_fit.imag[res_index], "ms", markersize=6)
    if plot_origin:
        ax0.axvline(0, color="gray", alpha=0.5)
        ax0.axvline(1, color="gray", alpha=0.5)
        ax0.axhline(0, color="gray", alpha=0.5)
    ax0.set_xlabel("Re[S21]")
    ax0.set_ylabel("Im[S21]")
    ax0.legend(["data", "init. fit", "best fit", f"S21({res_freq_name})"])

    ax1 = fig.add_subplot(gs[1, 1])
    ax1.plot(freqs / 1e6, np.abs(data), "-", color="tab:red")
    ax1.plot(freqs / 1e6, np.abs(init_fit), ".", color="tab:red", markersize=0.5)
    ax1.plot(freqs / 1e6, np.abs(best_fit), "--", color="tab:red")
    ax1.axvline(freqs[res_index] / 1e6, color="magenta", alpha=0.5)
    ax1.set_xlabel("Frequency (MHz)")
    ax1.set_ylabel("|S21|", color="tab:red")
    ax1.legend(["data", "init. fit", "best fit", res_freq_name])
    for i, leg_handle in enumerate(ax1.get_legend().legendHandles):
        if i != 3:
            leg_handle.set_color("black")

    ax2 = ax1.twinx()
    ax2.plot(freqs / 1e6, np.angle(data), "-", color="tab:blue")
    ax2.plot(freqs / 1e6, np.angle(best_fit), "--", color="tab:blue")
    ax2.plot(freqs / 1e6, np.angle(init_fit), ".", color="tab:blue", markersize=0.5)
    ax2.set_ylabel("Arg[S21] (rad)", color="tab:blue")

    ax3 = fig.add_subplot(gs[0, 1], sharex=ax1)
    ax3.plot(freqs / 1e6, mag_residuals, "k-")
    ax3.axvline(freqs[res_index] / 1e6, color="magenta", alpha=0.5)
    ax3.set_ylabel("Abs. Residuals" + (" (%)" if normalize_residuals else ""))
    setp(ax3.get_xticklabels(), visible=False)


def unwrap_phase(data, is_complex=True):
    """Returns continuous phase of complex data or angular data.

    Assumes data is ordered with relation to the frequencies swept
    to create it. Also assumes data is not noisy. If provided data is complex
    it assumes full I/Q complex data was provided. If provided data is real it
    assumes phases were provided.
    """
    # Finds correct phases of complex_data in range (-pi, pi)
    wrapped_phases = np.arctan2(np.imag(data), np.real(data)) if is_complex else data

    unwrapped_phases = wrapped_phases
    for i in range(unwrapped_phases.size - 1):
        phase_diff = unwrapped_phases[i + 1] - unwrapped_phases[i]
        # Check if phase jump is at this point
        if np.abs(phase_diff) > np.pi / 2:
            # Find out if phase jump was clockwise or counterclockwise
            diff_sign = np.sign(phase_diff)
            # Add or subtract 2*pi to all subsequent points
            unwrapped_phases[(i + 1) :] -= 2 * np.pi * diff_sign  # pep8:E203
    return unwrapped_phases


def algebraic_circle_fit(complex_data, verbose=False, eps=1e-7):
    """Algebraically find best fit parameterization of cdata to a circle.

    Uses Pratt's Approximation to Gradient-Weighted Algebraic Fits described
    in Chernov & Lesort, Journal of Mathematical Imaging and Vision, 2005,
    to fit a circle in the complex plane by mapping it to an eigenvalue
    problem. Uses scipy.linalg.eig for diagonalization.

    Parameters:
    -----------
    complex_data ([complex, ]): Iterable of complex data representing a
        (possibly noisy) circle.

    Keyword Arguments:
    ------------------
    verbose (bool): Whether or not to print summary of circle fit procedure.
    eps (float): Tolerance for considering a singular value to be zero in
        this algorithm's linear algebra calculations.

    Returns:
    --------
    [float, ]: List of [A, B, C, D] parameters describing a circle in the
        complex plane as: A(x^2 + y^2) + Bx + Cy + D = 0, constrained by
        B^2 + C^2 - 4AD = 1, where (x, y) = (Re[cdata], Im[cdata]).
        These correspond to the radius 'r' of the circle and its center
        position xc + 1j*yc as:
            xc = -B/2A
            yc = -C/2A
            r = 1/2|A|
    """
    x = np.real(complex_data)
    y = np.imag(complex_data)
    z = x * x + y * y  # Squared magnitude of complex data

    # Circular constraint matrix (eqn. 7 Probst)
    B = np.zeros((4, 4))
    B[1, 1] = B[2, 2] = 1
    B[0, 3] = B[3, 0] = -2

    # Calculate 'moments' or 'weights' of data:
    M = np.zeros((4, 4))
    circle_data = np.array([z, x, y, np.array([1] * len(complex_data))])
    M = circle_data @ circle_data.T

    # Solve generalized eigenvalue problem for smallest positive eigval
    # using Newton's method. This is guaranteed with x_guess=0 to return
    # the smallest non-negative eigenvalue (mentioned in Probst).
    eta = newton(lambda x: np.linalg.det(M - x * B), 0)
    _, s, Vh = svd(M - eta * B)
    vec = Vh[s <= eps, :].T
    normalization_param = vec.T @ B @ vec
    vec *= np.sign(normalization_param) / np.sqrt(np.abs(normalization_param))
    if verbose:
        print("eta = " + str(eta))
        print("eigenvector = " + str(vec))
        print("(M - eta*B)*eigenvector = " + str((M - eta * B) @ vec))
        print(
            "Constraint condition B^2+C^2-4AD = "
            + str(vec[1] ** 2 + vec[2] ** 2 - 4 * vec[0] * vec[3])
        )
    return vec


def circle_fit_square_diff(complex_data):
    """Sum of square deviations of data from an algebraically fit circle."""
    # Algebraically fit data to a circle
    fit_result = algebraic_circle_fit(complex_data)
    # Calculate radial deviations of data from circle center
    x_dev = np.real(complex_data) + 0.5 * fit_result[1] / fit_result[0]
    y_dev = np.imag(complex_data) + 0.5 * fit_result[2] / fit_result[0]
    r = 0.5 / np.abs(fit_result[0])

    # Compare distance of data points from fit circle center with fit radius
    return r**2 - x_dev * x_dev - y_dev * y_dev


def fit_cable_delay(
    freqs,
    complex_data,
    fit_offset=True,
    return_full_result=False,
    fixed_slope=None,
    verbose=True,
):
    """Find linear (in frequency) phase offsetting complex data."""
    # First find an initial guess of the cable delay (phase slope and offset), by
    # conducting a linear regression fit of the phase data
    phases = unwrap_phase(complex_data)
    freq_avg = np.mean(freqs)
    phase_avg = np.mean(phases)
    freq_deviations = freqs - freq_avg
    phase_slope = np.sum(freq_deviations * (phases - phase_avg))
    phase_slope /= np.dot(freq_deviations, freq_deviations)
    phase_offset = phase_avg - phase_slope * freq_avg

    # Initialize phase_slope/offset Parameter() using linear regression fit
    # result as guess
    params = lmfit.Parameters()
    params.add(
        "phase_slope",
        value=phase_slope if fixed_slope is None else fixed_slope,
        vary=fixed_slope is None,
        min=-1e-3,
        max=1e-3,
    )
    if fit_offset:
        params.add(
            "phase_offset",
            value=phase_offset,
            vary=True,
        )

    # Define function to be minimized
    # (deviations of data from fit circle edge)

    def residuals(parameters, freqs, complex_data):
        phase_slope = parameters["phase_slope"].value
        phase_offset = parameters["phase_offset"].value if fit_offset else 0
        unrolled_data = complex_data * np.exp(
            -1j * (phase_slope * freqs + phase_offset)
        )
        return circle_fit_square_diff(unrolled_data)

    fit_result = lmfit.minimize(residuals, params, args=(freqs, complex_data))
    if verbose:
        print(lmfit.fit_report(fit_result))
        phase_slope = fit_result.params["phase_slope"].value
        phase_offset = fit_result.params["phase_offset"].value if fit_offset else 0
        unrolled_data = complex_data * np.exp(
            -1j * (phase_slope * freqs + phase_offset)
        )
        _, axs = subplots(1, 2, figsize=(10, 5))
        ax = axs[0]
        ax.plot(freqs, unwrap_phase(complex_data), "k-")
        ax.plot(freqs, unwrap_phase(unrolled_data), "g-")
        ax.set_title("Unrolled data (green) and original data")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Unwrapped Phase (rad)")

        ax = axs[1]
        ax.plot(np.real(unrolled_data), np.imag(unrolled_data), "r-")
        ax.axhline(0)
        ax.axvline(0)
        ax.set_title("Unrolled data")
        ax.set_xlabel("Re[S21]")
        ax.set_ylabel("Im[S21]")
        show()
    if return_full_result:
        return fit_result
    elif fit_offset:
        return (
            fit_result.params["phase_slope"].value,
            fit_result.params["phase_offset"].value,
        )
    return fit_result.params["phase_slope"].value
