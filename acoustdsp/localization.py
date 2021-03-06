"""
Module which implements several acoustical sound source localization related
methods.

References
----------
[1] C. Knapp and G. Carter, “The generalized correlation method for estimation
    of time delay,” IEEE Trans. Acoust., Speech, Signal Process., vol. 24,
    no. 4, pp. 320-327, Aug. 1976, doi: 10.1109/TASSP.1976.1162830.
[2] Xiaoming Lai and H. Torp, “Interpolation methods for time-delay estimation
    using cross-correlation method for blood velocity measurement,” IEEE Trans.
    Ultrason., Ferroelect., Freq. Contr., vol. 46, no. 2, pp. 277-290, Mar.
    1999, doi: 10.1109/58.753016.
[3] Lei Zhang and Xiaolin Wu, “On Cross Correlation Based Discrete Time Delay
    Estimation,” in Proceedings. (ICASSP '05). IEEE International Conference on
    Acoustics, Speech, and Signal Processing, 2005., Philadelphia,
    Pennsylvania, USA, 2005, vol. 4, pp. 981-984.
    doi: 10.1109/ICASSP.2005.1416175.
[4] L. Svilainis, “Review on Time Delay Estimate Subsample Interpolation in
    Frequency Domain,” IEEE Trans. Ultrason., Ferroelect., Freq. Contr., vol.
    66, no. 11, pp. 1691-1698, Nov. 2019, doi: 10.1109/TUFFC.2019.2930661.

"""
import itertools
import warnings

import numba
import numpy as np

from .utils import cc_model


def gcc(sig: np.ndarray, refsig: np.ndarray,
        weighting: str = "direct") -> np.ndarray:
    """
    Compute the Generalized Cross-Correlation according to [1].

    Parameters
    ----------
    sig: np.ndarray
        Input signal, specified as an SxN matrix, with S being the number
        of signals of size N.
    refsig: np.ndarray
        Reference signal, specified as a column or row vector of size
        N.
    fs: int, optional
        Signal sampling rate in Hz, specified as a real-valued scalar.
        Defaults to 1.
    weighting: str, optional
        Define the weighting function for the generalized
        cross-correlation. Defaults to 'direct' weighting.
    Returns
    -------
    R: np.ndarray
        Cross-correlation between the input signal and the reference
        signal. `R` has a size of (2N-1) x S.
    """
    if (weighting.lower() != "direct" and weighting.lower() != "phat"):
        raise ValueError("This function currently only supports Direct and "
                         "PHAT weighting.")

    fft_len = sig.shape[0] + refsig.shape[0]

    SIG = np.fft.rfft(sig, n=fft_len, axis=0)
    REFSIG = np.fft.rfft(refsig, n=fft_len, axis=0)

    G = np.conj(REFSIG) * SIG   # Calculate Cross-Spectral Density
    W = np.abs(G) if weighting.lower() == "phat" else 1

    # Apply weighting and retrieve cross-correlation.
    R = np.fft.ifftshift(np.fft.irfft(G / W, n=fft_len, axis=0), axes=0)
    return R[1:, :]


def cc_parabolic_interp(R: np.ndarray, tdoa_region: np.ndarray, tau: float,
                        fs: int = 1):
    """
    Fit a parabolic function of the form: `ax^2 + bx + c` to the maximum
    value of a Cross-Correlation function. Returns the x-position of the
    vertex of the fitted parabolic function [2].

    Parameters
    ----------
    R: np.ndarray
        Input cross-correlation signal. R has a size of (2N-1) x S. Where
        S is the number of cross-correlations.
    tdoa_region: np.ndarray
        The region where the TDOA estimate is bounded to. This variable
        consists of a range of valid TDOA estimate indices.
    tau: float
        Estimated time delay value in seconds which maximizes the
        cross-correlation function `R`.
    fs: int, optional
        Signal sampling rate in Hz, specified as a real-valued scalar.
        Defaults to 1.
    Returns
    -------
    tau: float
        Improved time delay estimation in seconds.
    """
    if len(R.shape) < 2:
        R = np.atleast_2d(R).T

    max_indices = np.argmax(R[tdoa_region, :], axis=0) + tdoa_region[0]
    # Retrieve the values around the maximum of R
    y = np.array([R[idx - 1: idx + 2, i] for i, idx in enumerate(max_indices)])

    tau_imp = tau
    for i, y_i in enumerate(y):
        if len(y_i) == 3:
            # Perform parabolic interpolation and return the improved tau
            # value.
            d1 = y_i[1] - y_i[0]
            d2 = y_i[2] - y_i[0]
            a = -d1 + d2 / 2
            b = 2 * d1 - d2 / 2
            vertices = -b / (2 * a)
            # vertex - 1 is the sample-offset from maximum point of R
            tau_imp[i] += (vertices - 1) / fs
    return tau_imp


def cc_gaussian_interp(R: np.ndarray, tdoa_region: np.ndarray, tau: float,
                       fs: int = 1):
    """
    Fit a gaussian function of the form: `a * exp(-b(x - c)^2)` to the
    maximum value of a Cross-Correlation function. Returns the x-position
    of the vertex of the fitted gaussian function [3].

    Parameters
    ----------
    R: np.ndarray
        Input cross-correlation signal. R has a size of (2N-1) x S. Where
        S is the number of cross-correlations.
    tdoa_region: np.ndarray
        The region where the TDOA estimate is bounded to. This variable
        consists of a range of valid TDOA estimate indices.
    tau: float
        Estimated time delay value in seconds which maximizes the
        cross-correlation function `R`.
    fs: int, optional
        Signal sampling rate in Hz, specified as a real-valued scalar.
        Defaults to 1.
    Returns
    -------
    tau: float
        Improved time delay estimation in seconds.
    """
    if len(R.shape) < 2:
        R = np.atleast_2d(R).T
    max_indices = np.argmax(R[tdoa_region, :], axis=0) + tdoa_region[0]

    # Retrieve the values around the maximum of R. R needs to be positive for
    # indices around the maximum value. If this is not the case, take the
    # absolute value of the point (impacts fitting).
    y_ind = max_indices[:, np.newaxis] + np.arange(-1, 2)
    y_ind = np.where(y_ind < 0, 0, y_ind)
    y_ind = np.where(y_ind >= R.shape[0], R.shape[0] - 1, y_ind)
    y = R[y_ind, np.arange(0, y_ind.shape[0]).reshape(-1, 1)]

    if (y < 0).any():
        warnings.warn("Gaussian interpolation encountered negative R values. "
                      "Interpolation may not be correct.", RuntimeWarning)
        y = np.array([p + 2 * abs(np.min(p)) if (p < 0).any() else p
                      for p in y])

    c = (np.log(y[:, 2]) - np.log(y[:, 0])) / (4 * np.log(y[:, 1]) - 2
                                               * np.log(y[:, 0]) - 2
                                               * np.log(y[:, 2]))
    # vertex - 1 is the sample-offset from maximum point of R
    return tau + c / fs


def cc_sinc_interp(r: np.ndarray, tau: float, interp_mul: int, fs: int):
    """
    Fit a critically sampled sinc function to the maximum value of the
    cross-correlation function. Returns the improved time-delay found by the
    fitting.

        Parameters
    ----------
    R: np.ndarray
        Input cross-correlation signal. R has a size of (2N-1) x S. Where
        S is the number of cross-correlations.
    tau: float
        Estimated time delay value in seconds which maximizes the
        cross-correlation function `R`.
    interp_mul: int
        Interpolation factor equal to `T / T_i`. Where `T` is the sampling
        period of the original sampled signal. `T_i` is the interpolation
        sampling period.
    fs: int, optional
        Signal sampling rate in Hz, specified as a real-valued scalar.
        Defaults to 1.
    half_width: float
        interpolation half width of the sinc fitting. Specifies the maximum
        time-delay to fit the sinc funtion around the maximum of the
        cross-correlation function.
    Returns
    -------
    tau: float
        Improved time delay estimation in seconds.
    """
    if(interp_mul <= 0):
        raise ValueError("Interpolation multiplier has to be a strictly"
                         " positive integer.")
    if len(r.shape) == 1:
        r = np.atleast_2d(r).T

    max_ind = np.atleast_1d((tau * fs + r.shape[0] // 2)).astype(np.int32)

    # Search 1 sample around the direct path components
    search_area = tau + (np.arange(-interp_mul, interp_mul + 1).reshape(-1, 1)
                         / (interp_mul * fs))
    cost_vector = __cc_sinc_interp_helper__(r, search_area, max_ind, fs)

    minima = np.argmin(cost_vector, axis=0)
    return search_area[minima, np.arange(0, minima.shape[0])]


@numba.njit
def __cc_sinc_interp_helper__(r, search_area, max_ind, fs):
    cost_vector = np.zeros(search_area.shape)
    half_width = r.shape[0] // 2
    t = np.arange(-half_width, half_width + 1) / fs

    amplitudes = [r[i_max, i] for i, i_max in enumerate(max_ind)]

    for i, r_i in enumerate(r.T):
        for j, t_0 in enumerate(search_area[:, i]):
            sinc = amplitudes[i] * np.sinc(fs * (t - t_0))
            cost_vector[j, i] = np.sum(np.square(sinc - r_i))
    return cost_vector


def cc_whittaker_shannon_interp(r: np.ndarray, tau_est: float,
                                num_points: int = None,
                                interp_factor: int = 100, fs: int = 1):
    """
    Function under test
    """
    if num_points is None:
        num_points = r.shape[0]
    if len(r.shape) == 1:
        r = np.atleast_2d(r).T

    argmax = np.argmax(r, axis=0).astype(np.int32)
    windows = argmax + np.arange(-num_points // 2, num_points // 2
                                 ).reshape(-1, 1)

    n_interp = argmax + (np.arange(-interp_factor // 2, interp_factor // 2 + 1)
                         / interp_factor).reshape(-1, 1)
    t_interp = (n_interp - argmax) / fs

    reconst = np.zeros((interp_factor + 1, r.shape[1]))

    for idx, window in enumerate(windows.T):
        window -= np.min(window) if np.min(window) < 0 else 0
        window -= (np.max(window) - (r.shape[0] - 1)
                   if np.max(window) >= r.shape[0] else 0)

        reconst[:, idx] = np.sum(r[window, idx, np.newaxis]
                                 * np.sinc(n_interp[:, idx]
                                 - window.reshape(-1, 1)), axis=0)

    max_vals = np.argmax(reconst, axis=0)
    tau_delta = np.array([t_interp[val, idx] for idx, val
                          in enumerate(max_vals)]).T
    return tau_est + tau_delta


def cc_windowed_sinc_interp(r: np.ndarray, t0_est: float, t1_est: float,
                            interp_mul: int, fs: int):
    if(interp_mul <= 0):
        raise ValueError("Interpolation multiplier has to be a strictly"
                         " positive integer.")

    # Ensure `r` is a 2D Numpy array.
    if len(r.shape) == 1:
        r = np.atleast_2d(r).T

    L = (r.shape[0] + 1) / (2 * fs)
    t_corr = np.arange(-L * fs + 1, L * fs) / fs

    fs_res = fs * interp_mul

    # Search a single sample around tau_1 and tau_2
    search_area_t0 = t0_est + np.arange(-interp_mul, interp_mul + 1
                                        ).reshape(-1, 1) / (2 * fs_res)
    search_area_t1 = t1_est + np.arange(-interp_mul, interp_mul + 1
                                        ).reshape(-1, 1) / (2 * fs_res)

    models = np.zeros((t_corr.shape[0], search_area_t0.shape[0],
                       search_area_t1.shape[0]))
    for t0_idx, t0 in enumerate(search_area_t0):
        for t1_idx, t1 in enumerate(search_area_t1):
            models[:, t0_idx, t1_idx] = cc_model(t_corr, fs * np.pi, t0, t1, L)

    difference = models - r[:, :, np.newaxis]
    cost_vector = np.sum(np.square(difference), axis=0)

    minimum = np.min(cost_vector)
    est0_idx, est1_idx = np.where(cost_vector == minimum)

    return search_area_t0[est0_idx] - search_area_t1[est1_idx]


def calculate_tdoa(rirs: np.ndarray, mic_pairs: np.ndarray, max_td: int,
                   fs: int = 1, weighting: str = "direct",
                   interp: str = "None"):
    """
    Calculate the Time Difference of Arrival using the
    Generalized Cross-Correlation method.

    Parameters
    ----------
    rirs: np.ndarray
        Input Room Impulse Responses measured using the input
        mic_array. Shape of `rirs` needs to be equal to shape of
        `mic_array` (N x M), where N is the length of the microphone
        signal and M is the number of microphones.
    mic_pairs: np.ndarray
        A list containing all possible microphone pairs P in a given
        microphone-array setup. For M microphones, there are P * (P - 1) / 2
        unique pairs.
    max_td: int
        The maximum time in samples for a wavefront to propagate through
        the given microphone array configuration.
    fs: int, optional
        Signal sampling rate in Hz, specified as a real-valued scalar.
        Defaults to 1.
    weighting: str, optional
        Define the weighting function for the generalized
        cross-correlation. Defaults to 'direct' weighting.
    interp: str, optional
        Specify which interpolation method to use for improving the
        TDOA. Possible interpolation methods are: `None`, `Parabolic` or
        `Gaussian`. Defaults to `None`.
    Returns
    -------
    tau_hat: np.ndarray
        Time Difference of Arrival between all microphone pairs. The
        number of microphone pairs is: num_mics * (num_mics - 1) / 2
    """
    if (weighting.lower() not in ["direct", "phat"]):
        raise ValueError("This function currently only supports Direct and "
                         "PHAT weighting.")

    if (interp.lower() not in ["none", "parabolic", "gaussian"]):
        raise ValueError("This function currently only supports Parabolic and "
                         "Gaussian interpolation.")

    offset = rirs.shape[0]
    tdoa_region = offset + np.arange(-max_td, max_td + 1)

    # Estimate the time difference of arrival using GCC
    r = gcc(rirs[:, mic_pairs[:, 0]], rirs[:, mic_pairs[:, 1]])
    max_idices = np.argmax(r[tdoa_region, :], axis=0)
    tau_hat = (max_idices - max_td) / fs

    # Perform interpolation method
    if interp.lower() == "parabolic":
        tau_hat = cc_parabolic_interp(r, tdoa_region, tau_hat, fs)
    elif interp.lower() == "gaussian":
        tau_hat = cc_gaussian_interp(r, tdoa_region, tau_hat, fs)
    # Return estimated TDOA
    return tau_hat


def calc_tdoa_freq(rirs: np.ndarray, mic_array: np.ndarray, fs: int = 1):
    """
    Calculate the Time Difference of Arrival using the
    Generalized Cross-Correlation method in the frequency domain
    [4].

    Parameters
    ----------
    rirs: np.ndarray
        Input Room Impulse Responses measured using the input
        mic_array. Shape of `rirs` needs to be equal to shape of
        `mic_array` (N x M), where N is the length of the microphone
        signal and M is the number of microphones.
    mic_array: np.ndarray
        Microphone array carthesian coordinates (M x D), where M is
        the number of microphones and D is the number of dimensions.
    fs: int, optional
        Signal sampling rate in Hz, specified as a real-valued scalar.
        Defaults to 1.
    Returns
    -------
    tau_hat: np.ndarray
        Time Difference of Arrival between all microphone pairs. The
        number of microphone pairs is: num_mics * (num_mics - 1) / 2
    """

    mic_pairs = np.array(list(itertools.combinations(range(mic_array.shape[0]),
                                                     2)))
    # 1. Find max. point in Cross-Correlation function
    sigs = rirs[:, mic_pairs[:, 0]]
    refsigs = rirs[:, mic_pairs[:, 1]]

    # Calculate Cross-Spectral Density
    r = gcc(sigs, refsigs)

    shift = np.argmax(r, axis=0) - sigs.shape[0]

    # 2. Shift gcc with the rough TD estimate
    sigs_0 = np.array([np.roll(sig, -shift[idx])
                       for idx, sig in enumerate(sigs.T)]).T

    # 3. Transform the resulting CCF to the frequency domain
    G_0 = np.conj(np.fft.rfft(refsigs, axis=0)) * np.fft.rfft(sigs_0, axis=0)

    # 4. Find phase angle of the CCF and unwrap it
    phase = np.angle(G_0)
    freq = np.atleast_2d(np.arange(0, phase.shape[0]) * np.pi / fs).T

    # 5. Find ToF using the direct mean of the phase
    phase[-1] = np.nan  # remove nyquist freq. from calculation
    slope = np.nanmean(phase / freq, dtype='float64', axis=0)

    return (shift - slope) / fs


def calculate_doa(tau_hat: np.ndarray, V: np.ndarray):
    """
    Calculate the Direction Of Arrival by finding the slowness vector using
    Time Difference of Arrival estimation.

    Parameters
    ----------
    tau_hat: np.ndarray
        Estimated Time Difference of Arrival between all microphone pairs.
        The number of microphone pairs is: P = num_mics * (num_mics - 1) / 2.
        This vector has length (Px1)
    V: np.ndarray
        The sensor difference matrix in 3D space. V is an (P x D)
        matrix, with P number of microphone pairs and D physical
        dimensions.
    Returns
    -------
    doa: np.ndarray of floats
        Estimated Direction of Arrival in Carthesian coordinates (Dx1)
    """
    # Calculate slowness vector k
    k_hat = np.inner(np.linalg.pinv(V), tau_hat)
    # Calculate and return direction of arrival
    return k_hat if np.sum(k_hat, axis=0) == 0 else (-k_hat
                                                     / np.linalg.norm(k_hat, 2,
                                                                      axis=0))


def get_sensor_difference_matrix(mic_array):
    """
    Calculate the sensor difference matrix of a given microphone
    array. This is equal to the difference of all possible
    microphone pairs in the array.

    Parameters
    ----------
    mic_array: np.ndarray
        Microphone array carthesian coordinates (M x D), where M is
        the number of microphones and D is the number of dimensions.
    Returns
    -------
    V: np.ndarray
        The sensor difference matrix in 3D space. V is an (P x D)
        matrix, with P number of microphone pairs and D physical
        dimensions.

    """
    # All possible microphone pairs P
    mic_pairs = np.array(list(itertools.combinations(range(mic_array.shape[0]),
                                                     2)))
    # Define the sensor difference matrix in 3D space (P x D)
    V = mic_array[mic_pairs[:, 0], :] - mic_array[mic_pairs[:, 1], :]
    return V


def get_propagation_time(mic_array: np.ndarray, fs: int = 1, c: float = 343.):
    """
    Calculates the maximum time it takes for a wavefront to propagate
    through a given microphone array. The input of `mic_array` is a
    (M x D) matrix, where M is the number of microphones and D is the
    number of spatial dimensions.

    Parameters
    ----------
    mic_array: np.ndarray
        Microphone array carthesian coordinates (M x D), where M is
        the number of microphones and D is the number of dimensions.
    fs: int, optional
        Signal sampling rate in Hz, specified as a real-valued scalar.
        Defaults to 1.

    c: float, optional
        Speed of sound in meters per second. Defaults to 343.0 m/s.
    Returns
    -------
    max_td: int
        The maximum time in samples for a wavefront to propagate through
        the given microphone array configuration.
    """
    V = get_sensor_difference_matrix(mic_array)
    # Maximum distance and time delay
    max_distance = max(np.linalg.norm(V, 2, axis=1))
    # Maximum accepted time delay difference
    return np.ceil(max_distance / c * fs).astype(int)
