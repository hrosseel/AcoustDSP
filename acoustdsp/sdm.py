"""
Module which implements the Spatial Decomposition Method by Tervo et al.

References
----------
[1] S Tervo, J Pätynen and T Lokki: Spatial Decomposition Method for Room
    Impulse Responses. J. Audio Eng. Soc 61(1):1–13, 2013.

"""
import itertools

import numpy as np
from scipy.linalg import hankel

from . import localization as loc
from .utils import cart2sph


def spatial_decomposition_method(rirs: np.ndarray, mic_array: np.ndarray,
                                 ref_rir: np.ndarray, fs: int,
                                 threshold_db: float = 60,
                                 ref_pos: np.ndarray = None,
                                 win_size: int = None, win_type: str = "none",
                                 interp_method: str = "none",
                                 c: float = 343.):
    """
    Compute the Spatial Decomposition Method by Tervo et al. This method
    divides an input Room Impulse Response (RIR) in small short-time windows
    and computes the Direction of Arrival (DoA) for every short-time window.
    This function returns a DoA estimate for every sample in the reference RIR.

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
    ref_pos: np.ndarray
        Reference microphone position of size (D x 1). The reference position
        is typically located in the geometric center of the microphone array.
    fs: int
        Signal sampling rate in Hz, specified as a real-valued scalar.
        Defaults to 1.
    ref_rir: np.ndarray, optional
        Reference pressure signal of size (N x 1). This pressure signal
        has to be located in the geometric center of the microphone array.
    threshold_db: float, optional
        The specified correlation threshold in dB. Correlation values lower
        than this theshold will be omitted from the TDOA based DOA estimation.
        The default threshold is -60 dB.
    win_size: int, optional
        Define the window size of the SDM analysis in samples. This window
        size should be larger than the propagation time of a sound wave
        through the microphone array. If this variable is not set, the default
        window size equals the maximum propagation time + 8 samples.
    win_type: str, optional
        Define the window type that is used for each frame. Possible to choose
        between `none`, for no windowing, and `hanning` for a Hanning window.
        Default window is 'none'.
    interp_method: str, optional
        Interpolation method used to improve the Direction of Arrival estimate
        for each window. Possible interpolation schemes are: 'none', 'gaussian'
        , 'parabolic', and 'sinc'. Defaults to 'none' interpolation.
    c: float, optional
        Speed of sound in meters per second. Defaults to 343.0 m/s.
    Returns
    -------
    doa: np.ndarray of floats
        Returns the estimated Direction of Arrival in Carthesian coordinates
        (Nx3), where N is the number of samples in the input Room Impulse
        Response.
    """
    if isinstance(interp_method, str):
        interp_method = interp_method.lower()
    else:
        raise ValueError("Argument 'interp_method' should be of type 'str'.")
    if isinstance(win_type, str):
        win_type = win_type.lower()
    else:
        raise ValueError("Argument 'win_type' should be of type 'str'.")
    if ref_pos is None:
        ref_pos = np.mean(mic_array, axis=0)

    max_td = loc.get_propagation_time(mic_array, fs, c)
    V = loc.get_sensor_difference_matrix(mic_array)

    rir_size = rirs.shape[0]
    num_mics = mic_array.shape[0]

    win_size = win_size if win_size else max_td + 1
    if win_type == "none":
        win = np.ones(win_size)
    elif win_type == "hanning":
        win = np.hanning(win_size)
    else:
        raise ValueError("Parameter `win_type` must have the value `none` or "
                         "`hanning`.")

    num_frames = rir_size - win_size + 1
    frames = np.array([hankel(rirs[:num_frames, mic], rirs[-win_size:, mic]).T
                       for mic in range(num_mics)]).T

    # perform windowing
    frames = np.einsum("ijk, j -> ijk", frames, win)
    # Get all possible microphone pairs P
    mic_pairs = np.array(list(itertools.combinations(range(mic_array.shape[0]),
                                                     2)))
    doas = np.full((ref_rir.shape[0], 3), np.nan)

    for idx, frame in enumerate(frames):
        tdoa_region = win_size + np.arange(-max_td - 1, max_td)
        # Estimate the time difference of arrival using GCC
        r = loc.gcc(frame[:, mic_pairs[:, 0]], frame[:, mic_pairs[:, 1]])
        max_ind = np.argmax(r[tdoa_region, :], axis=0)

        # Make sure the correlations are sufficiently high w.r.t. the signal
        # amplitude. If too low, TDOA estimation could be inaccurate.
        tdoas = np.zeros((1, mic_pairs.shape[0])) * np.nan
        threshold = 10 ** (-abs(threshold_db) / 5)

        maxima = r[max_ind + tdoa_region[0], np.arange(0, max_ind.shape[0])]

        if (maxima > threshold).any():
            tau_hat = (max_ind - max_td) / fs
            if interp_method == "none":
                tdoas = tau_hat
            elif interp_method == "gaussian":
                tdoas = loc.cc_gaussian_interp(r, tdoa_region, tau_hat, fs)
            elif interp_method == "parabolic":
                tdoas = loc.cc_parabolic_interp(r, tdoa_region, tau_hat, fs)
            elif interp_method == "sinc":
                tdoas = loc.cc_sinc_interp(r, tau_hat, 1000, fs)
            elif interp_method == "whittaker":
                tdoas = loc.cc_whittaker_shannon_interp(r, tau_hat,
                                                        interp_factor=1000,
                                                        fs=fs)
            else:
                raise ValueError("Unknown interpolation scheme is used. "
                                 "Please choose between: 'gaussian', "
                                 "'parabolic', or 'sinc' interpolation.")

            doas[idx + win_size // 2, :] = loc.calculate_doa(tdoas, V).T
    return cart2sph(doas)
