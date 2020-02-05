import numpy as np

from alias.src.wave_function import wave_function_array, wave_function, d_wave_function_array, d_wave_function, \
    dd_wave_function_array, dd_wave_function, vcheck


def xi(x, y, coeff, qm, qu, dim):
    """
    xi(x, y, coeff, qm, qu, dim)

    Function returning position of intrinsic surface at position (x,y)

    Parameters
    ----------

    x:  float, array_like; shape=(nmol)
        Coordinate in x dimension
    y:  float, array_like; shape=(nmol)
        Coordinate in y dimension
    coeff:	float, array_like; shape=(n_waves**2)
        Optimised surface coefficients
    qm:  int
        Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
    qu:  int
        Upper limit of wave frequencies in Fouier Sum representing intrinsic surface
    dim:  float, array_like; shape=(3)
        XYZ dimensions of simulation cell

    Returns
    -------

    xi_z:  float, array_like; shape=(nmol)
        Positions of intrinsic surface in z dimension

    """

    n_waves = 2 * qm + 1

    if np.isscalar(x):
        u_array = (np.array(np.arange(n_waves**2) / n_waves, dtype=int) - qm)
        v_array = (np.array(np.arange(n_waves**2) % n_waves, dtype=int) - qm)
        wave_check = (u_array >= -qu) * (u_array <= qu) * (v_array >= -qu) * (v_array <= qu)
        indices = np.argwhere(wave_check).flatten()

        fuv = wave_function_array(x, u_array[indices], dim[0]) * wave_function_array(y, v_array[indices], dim[1])
        xi_z = np.sum(fuv * coeff[indices])
    else:
        xi_z = np.zeros(x.shape)
        for u in range(-qu, qu+1):
            for v in range(-qu, qu+1):
                j = (2 * qm + 1) * (u + qm) + (v + qm)
                xi_z += wave_function(x, u, dim[0]) * wave_function(y, v, dim[1]) * coeff[j]
    return xi_z


def dxy_dxi(x, y, coeff, qm, qu, dim):
    """
    dxy_dxi(x, y, qm, qu, coeff, dim)

    Function returning derivatives of intrinsic surface at position (x,y) wrt x and y

    Parameters
    ----------

    x:  float
        Coordinate in x dimension
    y:  float
        Coordinate in y dimension
    coeff:	float, array_like; shape=(n_waves**2)
        Optimised surface coefficients
    qm:  int
        Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
    qu:  int
        Upper limit of wave frequencies in Fouier Sum representing intrinsic surface
    dim:  float, array_like; shape=(3)
        XYZ dimensions of simulation cell

    Returns
    -------

    dx_dxi:  float
        Derivative of intrinsic surface in x dimension
    dy_dxi:  float
        Derivative of intrinsic surface in y dimension

    """

    n_waves = 2 * qm + 1

    if np.isscalar(x):
        u_array = (np.array(np.arange(n_waves**2) / n_waves, dtype=int) - qm)
        v_array = (np.array(np.arange(n_waves**2) % n_waves, dtype=int) - qm)
        wave_check = (u_array >= -qu) * (u_array <= qu) * (v_array >= -qu) * (v_array <= qu)
        indices = np.argwhere(wave_check).flatten()

        dx_dxi = d_wave_function_array(x, u_array[indices], dim[0]) * wave_function_array(y, v_array[indices], dim[1])
        dy_dxi = wave_function_array(x, u_array[indices], dim[0]) * d_wave_function_array(y, v_array[indices], dim[1])

        dx_dxi = np.sum(dx_dxi * coeff[indices])
        dy_dxi = np.sum(dy_dxi * coeff[indices])

    else:
        dx_dxi = np.zeros(x.shape)
        dy_dxi = np.zeros(x.shape)
        for u in range(-qu, qu+1):
            for v in range(-qu, qu+1):
                j = (2 * qm + 1) * (u + qm) + (v + qm)
                dx_dxi += d_wave_function(x, u, dim[0]) * wave_function(y, v, dim[1]) * coeff[j]
                dy_dxi += wave_function(x, u, dim[0]) * d_wave_function(y, v, dim[1]) * coeff[j]

    return dx_dxi, dy_dxi


def ddxy_ddxi(x, y, coeff, qm, qu, dim):
    """
    ddxy_ddxi(x, y, coeff, qm, qu, dim)

    Function returning second derivatives of intrinsic surface at position (x,y) wrt x and y

    Parameters
    ----------

    x:  float
        Coordinate in x dimension
    y:  float
        Coordinate in y dimension
    coeff:	float, array_like; shape=(n_waves**2)
        Optimised surface coefficients
    qm:  int
        Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
    qu:  int
        Upper limit of wave frequencies in Fouier Sum representing intrinsic surface
    dim:  float, array_like; shape=(3)
        XYZ dimensions of simulation cell

    Returns
    -------

    ddx_ddxi:  float
        Second derivative of intrinsic surface in x dimension
    ddy_ddxi:  float
        Second derivative of intrinsic surface in y dimension

    """

    n_waves = 2 * qm + 1

    if np.isscalar(x):
        u_array = (np.array(np.arange(n_waves**2) / n_waves, dtype=int) - qm)
        v_array = (np.array(np.arange(n_waves**2) % n_waves, dtype=int) - qm)
        wave_check = (u_array >= -qu) * (u_array <= qu) * (v_array >= -qu) * (v_array <= qu)
        indices = np.argwhere(wave_check).flatten()

        ddx_ddxi = dd_wave_function_array(x, u_array[indices], dim[0]) * wave_function_array(y, v_array[indices], dim[1])
        ddy_ddxi = wave_function_array(x, u_array[indices], dim[0]) * dd_wave_function_array(y, v_array[indices], dim[1])

        ddx_ddxi = np.sum(ddx_ddxi * coeff[indices])
        ddy_ddxi = np.sum(ddy_ddxi * coeff[indices])

    else:
        ddx_ddxi = np.zeros(x.shape)
        ddy_ddxi = np.zeros(x.shape)
        for u in range(-qu, qu+1):
            for v in range(-qu, qu+1):
                j = (2 * qm + 1) * (u + qm) + (v + qm)
                ddx_ddxi += dd_wave_function(x, u, dim[0]) * wave_function(y, v, dim[1]) * coeff[j]
                ddy_ddxi += wave_function(x, u, dim[0]) * dd_wave_function(y, v, dim[1]) * coeff[j]

    return ddx_ddxi, ddy_ddxi


def xi_var(coeff, qm, qu, dim):
    """
    xi_var(coeff, qm, qu, dim)

    Calculate average variance of surface heights

    Parameters
    ----------

    coeff:	float, array_like; shape=(n_frame, n_waves**2)
        Optimised surface coefficients
    qm:  int
        Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
    qu:  int
        Upper limit of wave frequencies in Fouier Sum representing intrinsic surface
    dim:  float, array_like; shape=(3)
        XYZ dimensions of simulation cell

    Returns
    -------

    calc_var: float
        Variance of surface heights across whole surface

    """

    nframe = coeff.shape[0]
    n_waves = 2 * qm +1
    nxy = 40

    u_array = np.array(np.arange(n_waves**2) / n_waves, dtype=int) - qm
    v_array = np.array(np.arange(n_waves**2) % n_waves, dtype=int) - qm
    wave_filter = (u_array >= -qu) * (u_array <= qu) * (v_array >= -qu) * (v_array <= qu)
    indices = np.argwhere(wave_filter).flatten()
    Psi = vcheck(u_array[indices], v_array[indices]) / 4.

    coeff_filter = coeff[:,:,indices]
    mid_point = len(indices) / 2

    av_coeff = np.mean(coeff_filter[:, :,mid_point], axis=0)
    av_coeff_2 = np.mean(coeff_filter**2, axis=(0, 1)) * Psi

    calc_var = np.sum(av_coeff_2) - np.mean(av_coeff**2, axis=0)

    return calc_var