import numpy as np
from scipy.special import sici


# Sine Integral function
def __Si__(x: np.ndarray or float): return sici(x)[0]
# Cosine Integral function
def __Ci__(x: np.ndarray or float): return sici(x)[1]


def __log_minus_ci__(arg: np.ndarray or float, w0: float):
    """
    Implements f(arg) = log(arg) - Ci(2 * w0 * arg),
    where 'Ci' is equal to the Cosine Integral function.

    This function is always real-valued.

    Parameters
    ----------
    arg: np.ndarray or float
        argument of log(arg) and Ci(2 * w0 * arg)
    w0: float
        frequency parameter
    Returns
    -------
    f(arg) = log(arg) - Ci(2 * w0 * arg): np.ndarray
    """
    arg = np.abs(np.atleast_1d(arg))
    return np.where(arg == 0, -np.euler_gamma - np.log(2 * w0),
                    np.log(arg) - __Ci__(2 * w0 * arg))


def __cosc__(arg):
    # trick to avoid infinity when t_arg == 0
    t_arg = np.where(arg == 0, 1.0e-16, arg)
    return np.cos(t_arg) / t_arg


# Cross-correlation model of windowed sinc functions
def cc_model(t, w0, t0, t1, L):
    # Process all negative time indices
    t_neg = np.atleast_1d(t)[t <= 0]
    I1 = 1 / (2*w0) * (__cosc__(w0 * (t_neg + (t1 - t0))) * (
        -__log_minus_ci__(t_neg + L/2 - t0, w0) 
        + __log_minus_ci__(-L/2 - t0, w0) + __log_minus_ci__(L/2-t1, w0)
        - __log_minus_ci__(-L/2-t_neg-t1, w0)
        ) + np.sinc(w0 * (t_neg + (t1 - t0)) / np.pi) * (
            -__Si__(2*w0*(-L/2-t0)) + __Si__(2*w0*(t_neg+L/2-t0))
            - __Si__(2*w0*(-L/2-t_neg-t1)) + __Si__(2*w0*(L/2 - t1))))

    # Check wether t_neg contains t0 - t1
    I1[t_neg == t0 - t1] = 1 / (w0 ** 2) * (
        (np.cos(2 * w0 * (L/2 - t1)) - 1) / (L - 2 * t1)
        + (np.cos(2 * w0 * (-L/2 - t0)) - 1) / (L + 2 * t0)
        + (-w0 * __Si__(2*w0*(-L/2 - t0)) + w0 * __Si__(2*w0*(L/2-t1))))

    # Process all positive time indices
    t_pos = np.atleast_1d(t)[t > 0]
    I2 = 1 / (2*w0) * (__cosc__(w0 * (t_pos + (t1 - t0))) * (
        __log_minus_ci__(t_pos-L/2-t0, w0) - __log_minus_ci__(L/2-t0, w0)
        - __log_minus_ci__(-L/2-t1, w0) + __log_minus_ci__(L/2-t_pos-t1, w0)
        ) + np.sinc(w0 * (t_pos + (t1 - t0)) / np.pi) * (
            -__Si__(2*w0 * (t_pos-L/2-t0)) + __Si__(2*w0 * (L/2-t0))
            - __Si__(2*w0 * (-L/2-t1)) + __Si__(2*w0 * (L/2-t_pos-t1))))

    # Check wether t_pos contains t0 - t1
    I2[t_pos == t0 - t1] = 1 / (w0 ** 2) * (
        (np.cos(2 * w0 * (L/2 - t0)) - 1) / (L - 2 * t0)
        + (np.cos(2 * w0 * (L/2 + t1)) - 1) / (L + 2 * t1)
        + (w0 * __Si__(2*w0*(L/2 - t0)) + w0 * __Si__(2*w0 * (L/2 + t1))))

    return (np.concatenate([I1, I2]))
