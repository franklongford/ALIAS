import numpy as np
from scipy import constants as con

from alias.src.wave_function import vcheck


def wave_arrays(qm):

    n_waves = 2 * qm + 1
    wave_range = np.arange(n_waves ** 2)

    u_array = np.array(wave_range / n_waves, dtype=int) - qm
    v_array = np.array(wave_range % n_waves, dtype=int) - qm

    return u_array, v_array


def calculate_frequencies(u_array, v_array, dim):

    q2 = (u_array ** 2 / dim[0] ** 2
          + v_array ** 2 / dim[1] ** 2)

    q2 *= 4 * np.pi ** 2
    q = np.sqrt(q2)

    return q, q2


def filter_frequencies(q, fourier):

    unique_q = np.unique(q)[1:]
    fourier_sum = np.zeros(unique_q.shape)
    fourier_count = np.zeros(unique_q.shape)

    for i, qi in enumerate(q):
        index = np.where(unique_q == qi)
        fourier_sum[index] += fourier[i]
        fourier_count[index] += 1

    av_fourier = fourier_sum / fourier_count

    return unique_q, av_fourier


def power_spectrum_coeff(coeff_2, qm, qu, dim):
    """
    Returns power spectrum of average surface coefficients,
    corresponding to the frequencies in q2_set

    Parameters
    ----------

    coeff_2:  float, array_like; shape=(n_waves**2)
        Square of optimised surface coefficients
    qm:  int
        Maximum number of wave frequencies in Fourier Sum
        representing intrinsic surface
    qu:  int
        Upper limit of wave frequencies in Fourier Sum
        representing intrinsic surface
    dim:  float, array_like; shape=(3)
        XYZ dimensions of simulation cell

    Returns
    -------
    unique_q:  float, array_like
        Set of frequencies for power spectrum
        histogram
    av_fourier:  float, array_like
        Power spectrum histogram of Fourier
        series coefficients

    """

    u_array, v_array = wave_arrays(qm)
    wave_mask = (
        (u_array >= -qu) * (u_array <= qu)
        * (v_array >= -qu) * (v_array <= qu)
    )
    indices = np.argwhere(wave_mask).flatten()

    q, q2 = calculate_frequencies(
        u_array[indices], v_array[indices], dim)

    fourier = coeff_2[indices] / 4 * vcheck(
        u_array[indices], v_array[indices])

    # Remove redundant frequencies
    unique_q, av_fourier = filter_frequencies(q, fourier)

    return unique_q, av_fourier


def surface_tension_coeff(coeff_2, qm, qu, dim, T):
    """
    Returns spectrum of surface tension, corresponding to the frequencies in q2_set

    Parameters
    ----------

    coeff_2:  float, array_like; shape=(n_waves**2)
        Square of optimised surface coefficients
    qm:  int
        Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
    qu:  int
        Upper limit of wave frequencies in Fouier Sum representing intrinsic surface
    dim:  float, array_like; shape=(3)
        XYZ dimensions of simulation cell
    T:  float
        Average temperature of simulation (K)

    Returns
    -------
    unique_q:  float, array_like
        Set of frequencies for power spectrum histogram
    av_gamma:  float, array_like
        Surface tension histogram of Fourier series frequencies

    """

    u_array, v_array = wave_arrays(qm)
    wave_mask = (
            (u_array >= -qu) * (u_array <= qu)
            * (v_array >= -qu) * (v_array <= qu)
    )
    indices = np.argwhere(wave_mask).flatten()

    q, q2 = calculate_frequencies(
        u_array[indices], v_array[indices], dim)

    int_A = dim[0] * dim[1] * q2 * coeff_2[indices] * vcheck(u_array[indices], v_array[indices]) / 4
    gamma = con.k * T * 1E23 / int_A

    unique_q, av_gamma = filter_frequencies(q, gamma)

    return unique_q, av_gamma


def intrinsic_area_coeff(coeff_2, qm, qu, dim):
    """
    intrinsic_area_coeff(coeff_2, qm, qu, dim)

    Calculate the intrinsic surface area spectrum from coefficients at resolution qu

    Parameters
    ----------

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

    int_A:  float
        Relative size of intrinsic surface area, compared to cell cross section XY
    """

    u_array, v_array = wave_arrays(qm)
    wave_mask = (
            (u_array >= -qu) * (u_array <= qu)
            * (v_array >= -qu) * (v_array <= qu)
    )
    indices = np.argwhere(wave_mask).flatten()

    q2 = np.pi**2  * vcheck(u_array[indices], v_array[indices]) * (u_array[indices]**2 / dim[0]**2 + v_array[indices]**2 / dim[1]**2)
    int_A = q2 * coeff_2[indices]
    int_A = 1 + 0.5 * np.sum(int_A)

    return int_A


def cw_gamma_sr(q, gamma, kappa): return gamma + kappa * q**2


def cw_gamma_lr(q, gamma, kappa0, l0): return gamma + q**2 * (kappa0 + l0 * np.log(q))


def get_frequency_set(qm, qu, dim):
    """
    get_frequency_set(qm, qu, dim)

    Returns set of unique frequencies in Fourier series

    Parameters
    ----------

    qm:  int
        Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
    dim:  float, array_like; shape=(3)
        XYZ dimensions of simulation cell

    Returns
    -------

    q_set:  float, array_like
        Set of unique frequencies
    q2_set:  float, array_like
        Set of unique frequencies to bin coefficients to
    """

    u_array, v_array = wave_arrays(qm)
    wave_mask = (
            (u_array >= -qu) * (u_array <= qu)
            * (v_array >= -qu) * (v_array <= qu)
    )
    indices = np.argwhere(wave_mask).flatten()

    q2 = 4 * np.pi**2 * (u_array[indices]**2 / dim[0]**2 + v_array[indices]**2 / dim[1]**2)
    q = np.sqrt(q2)

    q_set = np.unique(q)[1:]
    q2_set = np.unique(q2)[1:]

    return q_set, q2_set
