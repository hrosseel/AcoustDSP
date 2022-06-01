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

import localization as loc


def spatial_decomposition_method(rirs: np.ndarray, ref_rir: np.ndarray,
                                 mic_array: np.ndarray, fs: int,
                                 threshold_db: float = 60,
                                 win_size: int = None,
                                 win_type: str = "none",
                                 interp_method: str = "gaussian",
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
    ref_rir: np.ndarray
        Reference pressure signal of size (N x 1). This pressure signal
        has to be located in the geometric center of the microphone array.
    mic_array: np.ndarray
        Microphone array carthesian coordinates (M x D), where M is
        the number of microphones and D is the number of dimensions.
    fs: int
        Signal sampling rate in Hz, specified as a real-valued scalar.
        Defaults to 1.
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
        for each window. Possible interpolation schemes are: 'gaussian',
        'parabolic', and 'sinc'. Defaults to 'gaussian' interpolation.
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

    max_td, V = loc.get_propagation_time(mic_array, fs, c)
    rir_size = rirs.shape[0]
    num_mics = mic_array.shape[0]

    win_size = win_size if win_size else max_td + 8
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
        tdoa_region = win_size + np.arange(-max_td, max_td + 1)

        # Estimate the time difference of arrival using GCC
        r = loc.gcc(frame[:, mic_pairs[:, 0]], frame[:, mic_pairs[:, 1]])
        max_idices = np.argmax(r[tdoa_region, :], axis=0) - max_td

        # Make sure the correlations are sufficiently high w.r.t. the signal
        # amplitude. If too low, TDOA estimation could be inaccurate.
        tdoas = np.zeros((1, mic_pairs.shape[0])) * np.nan
        threshold = 10 ** (-abs(threshold_db) / 5)
        if (r[max_idices + win_size, :] > threshold).any():
            tau_hat = max_idices / fs

            if interp_method == "gaussian":
                tdoas = loc.cc_gaussian_interp(r, tdoa_region, tau_hat, fs)
            elif interp_method == "parabolic":
                tdoas = loc.cc_parabolic_interp(r, tdoa_region, tau_hat, fs)
            elif interp_method == "sinc":
                tdoas = loc.cc_sinc_interp(r, tau_hat, 50, fs, max_td / fs)
            else:
                raise ValueError("Unknown interpolation scheme is used. Please "
                                 "choose between: 'gaussian', 'parabolic', or "
                                 "'sinc' interpolation.")

        distance = (idx + win_size // 2) / fs * c
        doas[idx + win_size // 2, :] = distance * loc.calculate_doa(tdoas, V).T

    return doas
