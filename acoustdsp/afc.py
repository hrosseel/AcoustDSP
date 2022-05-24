"""
This module contains functions relating to the research domain of
Acoustic Feedback Cancellation.

References
----------
[1] T. van Waterschoot and M. Moonen, “Fifty Years of Acoustic Feedback
    Control: State of the Art and Future Challenges,” Proc. IEEE, vol. 99,
    no. 2, pp. 288-327, Feb. 2011, doi: 10.1109/JPROC.2010.2090998.

"""

import numpy as np
from scipy import signal


def max_stable_gain(loop_FIR: np.ndarray, n_fft: int = 512):
    """
    This function calculates the Maximum Stable Gain (MSG) of a given
    loop transfer function [1].

    This function is based upon MATLAB code authored by Toon van Waterschoot.
    """
    if (n_fft % 2):
        raise ValueError("n_fft should be even.")

    w, loop_tf = signal.freqz(loop_FIR, 1, n_fft)

    # Get phase resp. from loop freq. resp. and divide by 2pi
    loop_phase = np.unwrap(np.angle(loop_tf)) / (2 * np.pi)

    # Find all lower indices close to a multiple of 2π
    idx_lower = np.array([idx for idx, val in
                          enumerate(np.diff(np.ceil(loop_phase))) if val != 0])
    w_lower = w[idx_lower]
    phase_lower = loop_phase[idx_lower]

    # Find all upper indices close to a multiple of 2π
    idx_upper = idx_lower + 1
    w_upper = w[idx_upper]
    phase_upper = loop_phase[idx_upper]

    # Perform linear interpolation to find frequencies where phase shift
    # is multiple of 2π
    phase_multiple = np.ceil(phase_upper)  # Closest multiple of 2pi
    w_interp = (w_upper - np.abs(phase_upper - phase_multiple)
                * (w_upper - w_lower) / np.abs(phase_upper - phase_lower))

    # Frequency response only at phase shifts that are a multiple of 2pi
    _, loop_tf_interp = signal.freqz(loop_FIR, 1, w_interp)
    loop_magnitude_resp = abs(loop_tf_interp)
    msg_idx = np.argmax(loop_magnitude_resp)

    # Calculate the MSG
    msg = -20 * np.log10(loop_magnitude_resp[msg_idx])
    omega_msg = w_interp[msg_idx]

    return (msg, omega_msg)
