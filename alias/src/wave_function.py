import numpy as np


def check_uv(u, v):
    """
    Returns weightings for frequencies u and v for anisotropic surfaces
    """

    if abs(u) + abs(v) == 0:
        return 4.
    elif u * v == 0:
        return 2.
    return 1.


def wave_function(x, u, Lx):
    """
    Wave in Fouier sum
    """

    if u >= 0:
        return np.cos(2 * np.pi * u * x / Lx)
    return np.sin(2 * np.pi * abs(u) * x / Lx)


def d_wave_function(x, u, Lx):
    """
    d_wave_function(x, u, Lx)

    Derivative of wave in Fouier sum wrt x

    """

    coeff = 2 * np.pi / Lx

    if u >= 0:
        return - coeff * u * np.sin(coeff * u * x)
    return coeff * abs(u) * np.cos(coeff * abs(u) * x)


def dd_wave_function(x, u, Lx):
    """
    dd_wave_function(x, u, Lx)

    Second derivative of wave in Fouier sum wrt x

    """

    return - 4 * np.pi ** 2 * u ** 2 / Lx ** 2 * wave_function(x, u, Lx)


def wave_function_array(x, u_array, Lx):
    """
    wave_function_array(x, u_array, Lx)

    Returns numpy array of all waves in Fouier sum

    """

    q = 2 * np.pi * np.abs(u_array) * x / Lx

    cos_indicies = np.argwhere(u_array >= 0)
    sin_indicies = np.argwhere(u_array < 0)
    f_array = np.zeros(u_array.shape)
    f_array[cos_indicies] += np.cos(q[cos_indicies])
    f_array[sin_indicies] += np.sin(q[sin_indicies])

    return f_array


def d_wave_function_array(x, u_array, Lx):
    """
    d_wave_function_array(x, u_array, Lx)

    Returns numpy array of all derivatives of waves in Fouier sum

    """

    q = 2 * np.pi * np.abs(u_array) * x / Lx

    cos_indicies = np.argwhere(u_array >= 0)
    sin_indicies = np.argwhere(u_array < 0)
    f_array = np.zeros(u_array.shape)
    f_array[cos_indicies] -= np.sin(q[cos_indicies])
    f_array[sin_indicies] += np.cos(q[sin_indicies])
    f_array *= 2 * np.pi * np.abs(u_array) / Lx

    return f_array


def dd_wave_function_array(x, u_array, Lx):
    """
    dd_wave_function_array(x, u_array, Lx)
    Returns numpy array of all second derivatives of waves in Fouier sum

    """
    return - 4 * np.pi ** 2 * u_array ** 2 / Lx ** 2 * wave_function_array(x, u_array, Lx)
