"""
Module which implements several utility functions that are used throughout
this library.
"""
import numpy as np
from matplotlib.collections import QuadMesh
import matplotlib.pyplot as plt
from scipy import signal as sig
from typing import Tuple


def mag2db(signal: np.ndarray, input_mode: str = "amplitude") -> np.ndarray:
    """
    Convert magnitude to decibels.

    Parameters
    ----------
        signal: np.ndarray
            Input array, specified as scalar or vector.
        mode: {"amplitude", "power"}, optional
            Express input array as either `amplitude` or
            `power` measurement. Default input array is expressed as
            `amplitude`.
    Returns
    -------
        np.ndarray
            Magnitude measurement expressed in decibels.
    """
    scaling = 10 if input_mode == "power" else 20
    return scaling * np.log10(np.abs(signal))


def normalize(signal: np.ndarray) -> np.ndarray:
    """
    Scale an input array so that it bounds are between [-1, 1].

    Parameters
    ----------
        signal: np.ndarray
            Input signal.
    Returns
    -------
        np.ndarray
            Normalized input signal.
    """
    return signal / (np.max(np.abs(signal)))


def spectrogram(signal: np.ndarray, fs: int, win: str or tuple,
                win_length: int, ax: plt.axes = None,
                clims: Tuple[float, float] = (None, None)
                ) -> Tuple[QuadMesh, Tuple[float, float]]:
    """
    Calculate and draw the spectrogram of the input signal on the
    provided axis.

    Parameters
    ----------
        signal: np.ndarray
            Input signal, specified as a 1-D array.
        fs: int
            Sampling frequency `fs`.
        win: str or tuple
            Desired window to use. For more information, see
            scipy.signal.spectrogram documentation.
        ax: plt.axes, optional
            Matplotlib axes on which the spectrogram will be drawn on. If
            no axes is specified, the spectrogram will be drawn on the
            current axes.
        clims: tuple of floats, optional
            Specify minimum and maximum colormap value respectively, of the
            spectrogram.
    Returns
    -------
        matplotlib.collections.QuadMesh
            Quadrilateral mesh, used for drawing the colorbar of the
            spectrogram.
        tuple of floats (min, max)
            Colormap limits of the spectrogram.
    """
    if (ax is None):
        ax = plt.gca()
    f, t, Sxx = sig.spectrogram(signal, fs, win, win_length)
    pcm = ax.pcolormesh(t, f, 10 * np.log10(Sxx), vmin=clims[0], vmax=clims[1],
                        shading='auto')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    return pcm, pcm.get_clim()


def cart2sph(cart: np.ndarray):
    """
    Transform Cartesian coordinates to spherical coordinates in a 3D
    space.

    Parameters
    ----------
        cart: np.ndarray
            Cartesian coordinates in 3D space. Array of size (N x 3).
    Returns
    -------
        sph: np.ndarray
            Array of spherical coordinates, size (N x 3) consisting of:
                azimuth: float
                    Azimuth angle, in radians, measured from the positive
                    x-axis. Value ranges from [-pi, pi] as a scalar.
                elevation: float
                    Elevation angle, in radians, measured from the x-y plane.
                    Value ranges from [-pi / 2, pi / 2] as a scalar.
                r: float
                    Radius. The distance from the origin to a point located in
                    the x-y-z plane as a scalar.
    """
    if len(cart.shape) < 2:
        cart = np.atleast_2d(cart)
    if cart.shape[1] != 3:
        raise ValueError("Parameter `cart` must be of size (N, 3)")

    azi = np.arctan2(cart[:, 1], cart[:, 0])
    ele = np.arctan2(cart[:, 2], np.sqrt(cart[:, 0] ** 2 + cart[:, 1] ** 2))
    r = np.linalg.norm(cart, 2, axis=1)
    return np.stack([azi, ele, r], axis=1)


def sph2cart(azimuth: float or np.ndarray, elevation: float or np.ndarray,
             r: float or np.ndarray):
    """
    Transform spherical coordinates to Cartesian coordinates.

    Parameters
    -------
        azimuth: float or np.ndarray
            Azimuth angle, in radians, measured from the positive x-axis.
            Value ranges from [-pi, pi] as a scalar or np.ndarray of size
            (N x 1).
        elevation: float or np.ndarray
            Elevation angle, in radians, measured from the x-y plane.
            Value ranges from [-pi / 2, pi / 2] as a scalar or np.ndarray of
            size (N x 1).
        r: float or np.ndarray
            Radius. The distance from the origin to a point located in the
            x-y-z plane as a scalar or np.ndarray of size (N x 1).

    Returns
    ----------
        cart: np.ndarray
            Numpy array of size (N x 3). Axis 1 represents the Cartesian
            coordinates: x, y, z, respectively.
    """
    azimuth = np.atleast_1d(azimuth)
    elevation = np.atleast_1d(elevation)
    r = np.atleast_1d(r)

    if len(azimuth) == len(elevation) == len(r):
        x = r * np.cos(elevation) * np.cos(azimuth)
        y = r * np.cos(elevation) * np.sin(azimuth)
        z = r * np.sin(elevation)
        return np.vstack([x, y, z]).T
    else:
        raise ValueError("All input parameters must be of the same size.")


def rad2deg(angle: float or np.ndarray):
    """
    Convert angle in Radians to smallest angle in degrees.

    Parameters
    ----------
    angle: float or np.ndarray
        Angle in Radians
    Returns
    -------
    angle: float or np.ndarray
        Smallest angle in Degrees
    """
    return ((angle / np.pi * 180) + 180) % 360 - 180
