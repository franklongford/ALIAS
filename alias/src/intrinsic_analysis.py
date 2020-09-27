"""
***************** INTRINSIC ANALYSIS MODULE *********************

Calculates properties of intrinsic surfaces, based on output files of
intrinsic_surface_method.py

********************************************************************
Created 22/2/2018 by Frank Longford

Contributors: Frank Longford

Last modified 27/2/2018 by Frank Longford
"""
import os
import sys
import tables

import numpy as np

from alias.io.hdf5_io import (
    make_hdf5,
    load_hdf5,
    save_hdf5,
    shape_check_hdf5,
    frame_check_hdf5,
    mode_check_hdf5
)
from alias.io.numpy_io import load_npy
from alias.src.conversions import coeff_to_fourier_2
from alias.src.wave_function import (
    wave_function,
    d_wave_function,
    dd_wave_function
)

from .utilities import unit_vector, create_file_name


def make_pos_dxdy(xmol, ymol, coeff, nmol, dim, qm):
    """
    Calculate distances and derivatives at each molecular position with
    respect to intrinsic surface

    Parameters
    ----------

    xmol:  float, array_like; shape=(nmol)
        Molecular coordinates in x dimension
    ymol:  float, array_like; shape=(nmol)
        Molecular coordinates in y dimension
    coeff:	float, array_like; shape=(n_waves**2)
        Optimised surface coefficients
    nmol:  int
        Number of molecules in simulation
    dim:  float, array_like; shape=(3)
        XYZ dimensions of simulation cell
    qm:  int
        Maximum number of wave frequencies in Fourier Sum
        representing intrinsic surface

    Returns
    -------

    int_z_mol:  array_like (float); shape=(nframe, 2, qm+1, nmol)
        Molecular distances from intrinsic surface
    int_dxdy_mol:  array_like (float); shape=(nframe, 4, qm+1, nmol)
        First derivatives of intrinsic surface wrt x and y at xmol, ymol
    int_ddxddy_mol:  array_like (float); shape=(nframe, 4, qm+1, nmol)
        Second derivatives of intrinsic surface wrt x and y at xmol, ymol

    """

    int_z_mol = np.zeros((qm+1, 2, nmol))
    int_dxdy_mol = np.zeros((qm+1, 4, nmol))
    int_ddxddy_mol = np.zeros((qm+1, 4, nmol))

    tmp_int_z_mol = np.zeros((2, nmol))
    tmp_dxdy_mol = np.zeros((4, nmol))
    tmp_ddxddy_mol = np.zeros((4, nmol))

    for qu in range(qm+1):

        if qu == 0:
            j = (2 * qm + 1) * qm + qm
            f_x = wave_function(xmol, 0, dim[0])
            f_y = wave_function(ymol, 0, dim[1])

            tmp_int_z_mol[0] += f_x * f_y * coeff[0][j]
            tmp_int_z_mol[1] += f_x * f_y * coeff[1][j]

        else:
            for u in [-qu, qu]:
                for v in range(-qu, qu+1):
                    j = (2 * qm + 1) * (u + qm) + (v + qm)

                    f_x = wave_function(xmol, u, dim[0])
                    f_y = wave_function(ymol, v, dim[1])
                    df_dx = d_wave_function(xmol, u, dim[0])
                    df_dy = d_wave_function(ymol, v, dim[1])
                    ddf_ddx = dd_wave_function(xmol, u, dim[0])
                    ddf_ddy = dd_wave_function(ymol, v, dim[1])

                    tmp_int_z_mol[0] += f_x * f_y * coeff[0][j]
                    tmp_int_z_mol[1] += f_x * f_y * coeff[1][j]
                    tmp_dxdy_mol[0] += df_dx * f_y * coeff[0][j]
                    tmp_dxdy_mol[1] += f_x * df_dy * coeff[0][j]
                    tmp_dxdy_mol[2] += df_dx * f_y * coeff[1][j]
                    tmp_dxdy_mol[3] += f_x * df_dy * coeff[1][j]
                    tmp_ddxddy_mol[0] += ddf_ddx * f_y * coeff[0][j]
                    tmp_ddxddy_mol[1] += f_x * ddf_ddy * coeff[0][j]
                    tmp_ddxddy_mol[2] += ddf_ddx * f_y * coeff[1][j]
                    tmp_ddxddy_mol[3] += f_x * ddf_ddy * coeff[1][j]

            for u in range(-qu+1, qu):
                for v in [-qu, qu]:
                    j = (2 * qm + 1) * (u + qm) + (v + qm)

                    f_x = wave_function(xmol, u, dim[0])
                    f_y = wave_function(ymol, v, dim[1])
                    df_dx = d_wave_function(xmol, u, dim[0])
                    df_dy = d_wave_function(ymol, v, dim[1])
                    ddf_ddx = dd_wave_function(xmol, u, dim[0])
                    ddf_ddy = dd_wave_function(ymol, v, dim[1])

                    tmp_int_z_mol[0] += f_x * f_y * coeff[0][j]
                    tmp_int_z_mol[1] += f_x * f_y * coeff[1][j]
                    tmp_dxdy_mol[0] += df_dx * f_y * coeff[0][j]
                    tmp_dxdy_mol[1] += f_x * df_dy * coeff[0][j]
                    tmp_dxdy_mol[2] += df_dx * f_y * coeff[1][j]
                    tmp_dxdy_mol[3] += f_x * df_dy * coeff[1][j]
                    tmp_ddxddy_mol[0] += ddf_ddx * f_y * coeff[0][j]
                    tmp_ddxddy_mol[1] += f_x * ddf_ddy * coeff[0][j]
                    tmp_ddxddy_mol[2] += ddf_ddx * f_y * coeff[1][j]
                    tmp_ddxddy_mol[3] += f_x * ddf_ddy * coeff[1][j]

        int_z_mol[qu] += tmp_int_z_mol
        int_dxdy_mol[qu] += tmp_dxdy_mol
        int_ddxddy_mol[qu] += tmp_ddxddy_mol

    int_z_mol = np.swapaxes(int_z_mol, 0, 1)
    int_dxdy_mol = np.swapaxes(int_dxdy_mol, 0, 1)
    int_ddxddy_mol = np.swapaxes(int_ddxddy_mol, 0, 1)

    return int_z_mol, int_dxdy_mol, int_ddxddy_mol


def create_intrinsic_positions_dxdyz(
        directory, file_name, nmol, nframe, qm, n0,
        phi, dim, recon=0, ow_pos=False):
    """
    Calculate distances and derivatives at each molecular position
    with respect to intrinsic surface in simulation frame

    Parameters
    ----------

    directory:  str
        File path of directory of alias analysis.
    file_name:  str
        File name of trajectory being analysed.
    nmol:  int
        Number of molecules in simulation
    nframe:  int
        Number of frames in simulation trajectory
    qm:  int
        Maximum number of wave frequencies in Fourier Sum
        representing intrinsic surface
    n0:  int
        Maximum number of molecular pivots in intrinsic surface
    phi:  float
        Weighting factor of minimum surface area term in surface
        optimisation function
    dim:  float, array_like; shape=(3)
        XYZ dimensions of simulation cell
    recon:  bool (default=False)
        Whether to use surface reconstructe coefficients
    ow_pos:  bool (default=False)
        Whether to overwrite positions and derivatives (default=False)

    """

    print("\n--- Running Intrinsic Positions and Derivatives Routine ---\n")

    surf_dir = os.path.join(directory, 'surface')
    pos_dir = os.path.join(directory, 'pos')
    intpos_dir = os.path.join(directory, 'intpos')
    if not os.path.exists(intpos_dir):
        os.mkdir(intpos_dir)

    file_name_pos = create_file_name(
        [file_name, qm, n0, int(1/phi + 0.5), nframe]
    )
    file_name_coeff = file_name_pos

    if recon:
        file_name_coeff += '_r'
        file_name_pos += '_r'

    intpos_data_file = os.path.join(intpos_dir + file_name_pos)

    if not os.path.exists(intpos_data_file + "_int_z_mol.hdf5"):
        make_hdf5(
            intpos_data_file + '_int_z_mol',
            (2, qm+1, nmol), tables.Float64Atom())
        make_hdf5(
            intpos_data_file + '_int_dxdy_mol',
            (4, qm+1, nmol), tables.Float64Atom())
        make_hdf5(
            intpos_data_file + '_int_ddxddy_mol',
            (4, qm+1, nmol), tables.Float64Atom())
        file_check = False

    elif not ow_pos:
        "Checking number of frames in current distance files"
        try:
            file_check = shape_check_hdf5(
                intpos_data_file + '_int_z_mol',
                (nframe, 2, qm+1, nmol))
            file_check *= shape_check_hdf5(
                intpos_data_file + '_int_dxdy_mol',
                (nframe, 4, qm+1, nmol))
            file_check *= shape_check_hdf5(
                intpos_data_file + '_int_ddxddy_mol',
                (nframe, 4, qm+1, nmol))
        except FileNotFoundError:
            file_check = False
    else:
        file_check = False

    pos_data_file = os.path.join(pos_dir + file_name)
    if not file_check:
        xmol = load_npy(
            pos_data_file + f'_{nframe}_xmol',
            frames=range(nframe))
        ymol = load_npy(
            pos_data_file + f'_{nframe}_ymol',
            frames=range(nframe))

        for frame in range(nframe):

            "Checking number of frames in int_z_mol file"
            frame_check_int_z_mol = frame_check_hdf5(
                intpos_data_file + '_int_z_mol', frame)
            frame_check_int_dxdy_mol = frame_check_hdf5(
                intpos_data_file + '_int_dxdy_mol', frame)
            frame_check_int_ddxddy_mol = frame_check_hdf5(
                intpos_data_file + '_int_ddxddy_mol', frame)

            mode_int_z_mol = mode_check_hdf5(
                frame_check_int_z_mol, ow_pos)
            mode_int_dxdy_mol = mode_check_hdf5(
                frame_check_int_dxdy_mol, ow_pos)
            mode_int_ddxddy_mol = mode_check_hdf5(
                frame_check_int_ddxddy_mol, ow_pos)

            check = mode_int_z_mol or mode_int_dxdy_mol or mode_int_ddxddy_mol
            if not check:
                sys.stdout.write(
                    "Calculating molecular distances "
                    f"and derivatives: frame {frame}\r"
                )
                sys.stdout.flush()

                surf_data_file = os.path.join(surf_dir, file_name_coeff)
                coeff = load_hdf5(surf_data_file + '_coeff', frame)

                int_z_mol, int_dxdy_mol, int_ddxddy_mol = make_pos_dxdy(
                    xmol[frame], ymol[frame], coeff, nmol, dim, qm)
                save_hdf5(intpos_data_file + '_int_z_mol',
                          int_z_mol, frame, mode_int_z_mol)
                save_hdf5(intpos_data_file + '_int_dxdy_mol',
                          int_dxdy_mol, frame, mode_int_dxdy_mol)
                save_hdf5(intpos_data_file + '_int_ddxddy_mol',
                          int_ddxddy_mol, frame, mode_int_ddxddy_mol)


def make_int_mol_count(zmol, int_z_mol, nslice, qm, dim):
    """
    Creates density histogram

    Parameters
    ----------

    zmol:  float, array_like; shape=(nmol)
        Molecular coordinates in z dimension
    int_z_mol:  array_like (float); shape=(nframe, 2, qm+1, nmol)
        Molecular distances from intrinsic surface
    nmol:  int
        Number of molecules in simulation
    nslice: int
        Number of bins in density histogram along axis
        normal to surface
    qm:  int
        Maximum number of wave frequencies in Fouier Sum
         representing intrinsic surface
    dim:  float, array_like; shape=(3)
        XYZ dimensions of simulation cell

    Returns
    -------

    mol_count_array:  int, array_like; shape=(qm+1, nslice, nz)
        Number histogram binned by molecular position along
         z axis and mean curvature H across qm resolutions

    """

    mol_count_array = np.zeros((qm+1, nslice))

    for qu in range(qm+1):

        temp_mol_count_array = np.zeros((nslice))

        int_z1 = int_z_mol[0][qu]
        int_z2 = int_z_mol[1][qu]

        z1 = zmol - int_z1 + dim[2]
        z2 = -(zmol - int_z2) + dim[2]

        z1 -= dim[2] * np.array(z1 / dim[2], dtype=int)
        z2 -= dim[2] * np.array(z2 / dim[2], dtype=int)

        temp_mol_count_array += np.histogram(
            z1, bins=nslice, range=[0, dim[2]])[0]
        temp_mol_count_array += np.histogram(
            z2, bins=nslice, range=[0, dim[2]])[0]

        mol_count_array[qu] += temp_mol_count_array

    return mol_count_array


def den_curve_hist(zmol, int_z_mol, int_ddxddy_mol, nslice, nz, qm, dim,
                   max_H=12):
    """
    Creates density and mean curvature histograms

    Parameters
    ----------

    zmol:  float, array_like; shape=(nmol)
        Molecular coordinates in z dimension
    int_z_mol:  array_like (float); shape=(nframe, 2, qm+1, nmol)
        Molecular distances from intrinsic surface
    int_ddxddy_mol:  array_like (float); shape=(nframe, 4, qm+1, nmol)
        Second derivatives of intrinsic surface wrt x and y at xmol, ymol
    nslice: int
        Number of bins in density histogram along axis normal to surface
    nz: int (optional)
        Number of bins in curvature histogram along axis normal to
        surface (default=100)
    qm:  int
        Maximum number of wave frequencies in Fouier Sum representing
        intrinsic surface
    dim:  float, array_like; shape=(3)
        XYZ dimensions of simulation cell

    Returns
    -------

    count_corr_array:  int, array_like; shape=(qm+1, nslice, nz)
        Number histogram binned by molecular position along z axis and
        mean curvature H across qm resolutions

    """

    count_corr_array = np.zeros((qm+1, nslice, nz))

    for qu in range(qm+1):

        temp_count_corr_array = np.zeros((nslice, nz))

        int_z1 = int_z_mol[0][qu]
        int_z2 = int_z_mol[1][qu]

        z1 = zmol - int_z1
        z2 = zmol - int_z2

        z1 -= dim[2] * np.array(2 * z1 / dim[2], dtype=int)
        z2 -= dim[2] * np.array(2 * z2 / dim[2], dtype=int)

        ddzx1 = int_ddxddy_mol[0][qu]
        ddzy1 = int_ddxddy_mol[1][qu]
        ddzx2 = int_ddxddy_mol[2][qu]
        ddzy2 = int_ddxddy_mol[3][qu]

        H1 = abs(ddzx1 + ddzy1)
        H2 = abs(ddzx2 + ddzy2)

        temp_count_corr_array += np.histogram2d(
            z1, H1, bins=[nslice, nz],
            range=[[-dim[2]/2, dim[2]/2], [0, max_H]])[0]
        temp_count_corr_array += (np.histogram2d(
            z2, H2, bins=[nslice, nz],
            range=[[-dim[2]/2, dim[2]/2], [0, max_H]])[0])[::-1]

        count_corr_array[qu] += temp_count_corr_array

    return count_corr_array


def create_intrinsic_den_curve_hist(directory, file_name, qm, n0, phi, nframe,
                                    nslice, dim,
                                    nz=100, recon=False, ow_hist=False):
    """
    Calculate density and curvature histograms across surface

    Parameters
    ----------

    directory:  str
        File path of directory of alias analysis.
    file_name:  str
        File name of trajectory being analysed.
    qm:  int
        Maximum number of wave frequencies in Fouier Sum representing intrinsic
         surface
    n0:  int
        Maximum number of molecular pivots in intrinsic surface
    phi:  float
        Weighting factor of minimum surface area term in surface optimisation
         function
    nframe:  int
        Number of frames in simulation trajectory
    nslice: int
        Number of bins in density histogram along axis normal to surface
    dim:  float, array_like; shape=(3)
        XYZ dimensions of simulation cell
    nz: int (optional)
        Number of bins in curvature histogram along axis normal to surface
         (default=100)
    recon:  bool (optional)
        Whether to use surface reconstructe coefficients (default=False)
    ow_hist:  bool (optional)
        Whether to overwrite density and curvature distributions
        (default=False)
    """

    print("\n--- Running Intrinsic Density and Curvature Routine --- \n")

    pos_dir = os.path.join(directory, 'pos')
    intpos_dir = os.path.join(directory, 'intpos')
    intden_dir = os.path.join(directory, 'intden')
    if not os.path.exists(intden_dir):
        os.mkdir(intden_dir)

    file_name_pos = create_file_name(
        [file_name, qm, n0, int(1./phi + 0.5), nframe])
    file_name_hist = create_file_name(
        [file_name, nslice, nz, qm, n0, int(1./phi + 0.5), nframe])

    if recon:
        file_name_pos += '_r'
        file_name_hist += '_r'

    intden_data_file = os.path.join(intden_dir + file_name_hist)
    if not os.path.exists(intden_data_file + '_count_corr.hdf5'):
        make_hdf5(intden_data_file + '_count_corr',
                  (qm+1, nslice, nz), tables.Float64Atom())
        file_check = False

    elif not ow_hist:
        "Checking number of frames in current distribution files"
        try:
            file_check = shape_check_hdf5(
                intden_data_file + '_count_corr', (nframe, qm+1, nslice, nz)
            )
        except FileNotFoundError:
            file_check = False
    else:
        file_check = False

    if not file_check:
        pos_data_file = os.path.join(pos_dir + file_name)
        zmol = load_npy(pos_data_file + '_{}_zmol'.format(nframe))
        COM = load_npy(pos_data_file + '_{}_com'.format(nframe))
        nmol = zmol.shape[1]
        com_tile = np.moveaxis(
            np.tile(COM, (nmol, 1, 1)), [0, 1, 2], [2, 1, 0])[2]
        zmol = zmol - com_tile

        for frame in range(nframe):

            "Checking number of frames in hdf5 files"
            frame_check_count_corr = frame_check_hdf5(
                intden_data_file + '_count_corr', frame)
            mode_count_corr = mode_check_hdf5(
                frame_check_count_corr, ow_hist)

            if mode_count_corr:
                sys.stdout.write(
                    "Calculating position and curvature "
                    "distributions: frame {}\r".format(frame))
                sys.stdout.flush()

                intpos_data_file = os.path.join(intpos_dir + file_name_pos)

                int_z_mol = load_hdf5(
                    intpos_data_file + '_int_z_mol', frame)
                int_ddxddy_mol = load_hdf5(
                    intpos_data_file + '_int_ddxddy_mol', frame)

                count_corr_array = den_curve_hist(
                    zmol[frame], int_z_mol, int_ddxddy_mol,
                    nslice, nz, qm, dim)
                save_hdf5(
                    intden_data_file + '_count_corr',
                    count_corr_array, frame, mode_count_corr)


def av_intrinsic_distributions(directory, file_name, dim, nslice, qm, n0, phi,
                               nframe, nsample,
                               nz=100, recon=False, ow_dist=False):
    """
    Summate average density and curvature distributions

    Parameters
    ----------

    directory:  str
        File path of directory of alias analysis.
    file_name:  str
        File name of trajectory being analysed.
    dim:  float, array_like; shape=(3)
        XYZ dimensions of simulation cell
    nslice: int
        Number of bins in density histogram along axis normal to surface
    qm:  int
        Maximum number of wave frequencies in Fouier Sum representing
        intrinsic surface
    n0:  int
        Maximum number of molecular pivots in intrinsic surface
    phi:  float
        Weighting factor of minimum surface area term in surface
        optimisation function
    nframe:  int
        Number of frames in simulation trajectory
    nsample:  int
        Number of frames to average over
    nz: int (optional)
        Number of bins in curvature histogram along axis normal to
        surface (default=100)
    recon:  bool (optional)
        Whether to use surface reconstructe coefficients (default=False)
    ow_dist:  bool (optional)
        Whether to overwrite average density and curvature
        distributions (default=False)

    Returns
    -------

    int_den_curve_matrix:  float, array_like; shape=(qm+1, nslice, nz)
        Average intrinsic density-curvature distribution for each
        resolution across nsample frames
    int_density:  float, array_like; shape=(qm+1, nslice)
        Average intrinsic density distribution for each resolution
        across nsample frames
    int_curvature:  float, array_like; shape=(qm+1, nz)
        Average intrinsic surface curvature distribution for each
        resolution across nsample frames

    """

    intden_dir = os.path.join(directory, 'intden')
    file_name_hist = create_file_name(
        [file_name, nslice, nz, qm, n0, int(1./phi + 0.5), nframe])
    file_name_dist = create_file_name(
        [file_name, nslice, nz, qm, n0, int(1. / phi + 0.5), nsample])

    if recon:
        file_name_hist += '_r'
        file_name_dist += '_r'

    count_data_file = os.path.join(intden_dir, file_name_hist)
    curve_data_file = os.path.join(intden_dir, file_name_dist)

    if not os.path.exists(curve_data_file + '_int_den_curve.npy') or ow_dist:

        int_den_curve_matrix = np.zeros((qm+1, nslice, nz))

        print("\n--- Loading in Density and Curvature Distributions ---\n")

        lslice = dim[2] / nslice
        Vslice = dim[0] * dim[1] * lslice

        for frame in range(nsample):
            sys.stdout.write("Frame {}\r".format(frame))
            sys.stdout.flush()

            count_corr_array = load_hdf5(
                count_data_file + '_count_corr', frame)
            int_den_curve_matrix += count_corr_array / (nsample * Vslice)

        np.save(curve_data_file + "_int_den_curve.npy", int_den_curve_matrix)

    else:
        int_den_curve_matrix = load_npy(curve_data_file + '_int_den_curve')

    int_density = np.sum(
        int_den_curve_matrix, axis=2) / 2.
    int_curvature = np.sum(
        np.moveaxis(int_den_curve_matrix, 1, 2), axis=2) / 2.

    return int_den_curve_matrix, int_density, int_curvature


def coeff_slice(coeff, qm, qu):
    """
    coeff_slice(coeff, qm, qu)

    Truncates coeff array up to qu resolution
    """

    n_waves_qm = 2 * qm + 1
    n_waves_qu = 2 * qu + 1

    index_1 = qm - qu
    index_2 = index_1 + n_waves_qu

    coeff_matrix = np.reshape(coeff, (n_waves_qm, n_waves_qm))
    coeff_qu = coeff_matrix[
        [slice(index_1, index_2) for _ in coeff_matrix.shape]
    ].flatten()

    return coeff_qu


def xy_correlation(coeff_2, qm, qu, dim):
    """
    xy_correlation(coeff_2, qm, qu, dim)

    Return correlation across xy plane using Wiener-Khinchin theorem

    Parameters
    ----------

    coeff_2:  float, array_like; shape=(n_waves**2)
        Square of optimised surface coefficients
    qm:  int
        Maximum number of wave frequencies in Fouier Sum representing
        intrinsic surface
    qu:  int
        Upper limit of wave frequencies in Fouier Sum representing
        intrinsic surface

    Returns
    -------

    xy_corr:  float, array_like; shape=(n_waves_qu**2)
        Length correlation function across xy plane

    """

    coeff_2[len(coeff_2)/2] = 0
    coeff_2_slice = coeff_slice(coeff_2, qm, qu)

    xy_corr, frequencies = coeff_to_fourier_2(coeff_2_slice, qu, dim)
    # xy_corr = np.abs(amplitudes_2) / np.mean(amplitudes_2)

    return xy_corr, frequencies


def make_den_curve(zmol, int_z_mol, int_dxdy_mol, nmol, nslice, nz, qm, dim):
    """
    Creates density and curvature distributions normal to surface

    Parameters
    ----------
    zmol:  float, array_like; shape=(nmol)
        Molecular coordinates in z dimension
    int_z_mol:  array_like (float); shape=(nframe, 2, qm+1, nmol)
        Molecular distances from intrinsic surface
    int_dxdy_mol:  array_like (float); shape=(nframe, 4, qm+1, nmol)
        First derivatives of intrinsic surface wrt x and y at xmol, ymol
    nmol:  int
        Number of molecules in simulation
    nslice: int
        Number of bins in density histogram along axis normal to surface
    nz: int (optional)
        Number of bins in curvature histogram along axis normal to
        surface (default=100)
    qm:  int
        Maximum number of wave frequencies in Fouier Sum representing
        intrinsic surface
    dim:  float, array_like; shape=(3)
        XYZ dimensions of simulation cell

    Returns
    -------

    count_corr_array:  int, array_like; shape=(qm+1, nslice, nz)
        Number histogram binned by molecular position along z axis
        and mean curvature H across qm resolutions

    """

    count_corr_array = np.zeros((qm+1, nslice, nz))

    for qu in range(qm+1):

        temp_count_corr_array = np.zeros((nslice, nz))

        int_z1 = int_z_mol[0][qu]
        int_z2 = int_z_mol[1][qu]

        z1 = zmol - int_z1
        z2 = -zmol + int_z2

        dzx1 = int_dxdy_mol[0][qu]
        dzy1 = int_dxdy_mol[1][qu]
        dzx2 = int_dxdy_mol[2][qu]
        dzy2 = int_dxdy_mol[3][qu]

        index1_mol = np.array(
            (z1 + dim[2]/2.) * nslice / dim[2], dtype=int) % nslice
        index2_mol = np.array(
            (z2 + dim[2]/2.) * nslice / dim[2], dtype=int) % nslice

        normal1 = unit_vector(np.array([-dzx1, -dzy1, np.ones(nmol)]))
        normal2 = unit_vector(np.array([-dzx2, -dzy2, np.ones(nmol)]))

        index1_nz = np.array(abs(normal1[2]) * nz, dtype=int) % nz
        index2_nz = np.array(abs(normal2[2]) * nz, dtype=int) % nz

        temp_count_corr_array += np.histogram2d(
            index1_mol, index1_nz,
            bins=[nslice, nz], range=[[0, nslice], [0, nz]])[0]
        temp_count_corr_array += np.histogram2d(
            index2_mol, index2_nz,
            bins=[nslice, nz], range=[[0, nslice], [0, nz]])[0]

        count_corr_array[qu] += temp_count_corr_array

    return count_corr_array


def create_intrinsic_den_curve_dist(directory, file_name, qm, n0, phi, nframe,
                                    nslice, dim,
                                    nz=100, recon=0, ow_hist=False):
    """
    Calculate density and curvature distributions across surface

    Parameters
    ----------

    directory:  str
        File path of directory of alias analysis.
    file_name:  str
        File name of trajectory being analysed.
    qm:  int
        Maximum number of wave frequencies in Fouier Sum representing
        intrinsic surface
    n0:  int
        Maximum number of molecular pivots in intrinsic surface
    phi:  float
        Weighting factor of minimum surface area term in surface optimisation
        function
    nframe:  int
        Number of frames in simulation trajectory
    nslice: int
        Number of bins in density histogram along axis normal to
        surface
    dim:  float, array_like; shape=(3)
        XYZ dimensions of simulation cell
    nz: int (optional)
        Number of bins in curvature histogram along axis normal to
        surface (default=100)
    recon:  bool (optional)
        Whether to use surface reconstructe coefficients (default=False)
    ow_count:  bool (optional)
        Whether to overwrite density and curvature distributions
        (default=False)
    """

    print("\n--- Running Intrinsic Density and Curvature Routine --- \n")

    pos_dir = os.path.join(directory, 'pos')
    intpos_dir = os.path.join(directory, 'intpos')
    intden_dir = os.path.join(directory, 'intden')

    if not os.path.exists(intden_dir):
        os.mkdir(intden_dir)

    file_name_pos = create_file_name(
        [file_name, qm, n0, int(1./phi + 0.5), nframe]
    )
    file_name_coeff = file_name_pos
    file_name_hist = create_file_name(
        [file_name, nslice, nz, qm, n0, int(1./phi + 0.5), nframe])

    if recon:
        file_name_pos += '_r'
        file_name_hist += '_r'
        file_name_coeff += '_r'

    count_data_file = os.path.join(intden_dir, file_name_hist)

    if not os.path.exists(count_data_file + '_count_corr.hdf5'):
        make_hdf5(count_data_file + '_count_corr',
                  (qm+1, nslice, nz), tables.Float64Atom())
        file_check = False

    elif not ow_hist:
        "Checking number of frames in current distribution files"
        try:
            file_check = shape_check_hdf5(
                count_data_file + '_count_corr',
                (nframe, qm+1, nslice, nz))
        except FileNotFoundError:
            file_check = False
    else:
        file_check = False

    if not file_check:
        pos_data_file = os.path.join(pos_dir + file_name)
        zmol = load_npy(pos_data_file + '_{}_zmol'.format(nframe))
        COM = load_npy(pos_data_file + '_{}_com'.format(nframe))
        nmol = zmol.shape[1]
        com_tile = np.moveaxis(
            np.tile(COM, (nmol, 1, 1)), [0, 1, 2], [2, 1, 0])[2]
        zmol = zmol - com_tile

        for frame in range(nframe):

            "Checking number of frames in hdf5 files"
            frame_check_count_corr = frame_check_hdf5(
                count_data_file + '_count_corr', frame)
            mode_count_corr = mode_check_hdf5(frame_check_count_corr, ow_hist)

            if mode_count_corr:
                sys.stdout.write(
                    "Calculating position and curvature distributions:"
                    f" frame {frame}\r")
                sys.stdout.flush()

                intpos_data_file = os.path.join(intpos_dir + file_name_pos)
                int_z_mol = load_hdf5(intpos_data_file + '_int_z_mol', frame)
                int_dxdy_mol = load_hdf5(
                    intpos_data_file + '_int_dxdy_mol', frame)

                count_corr_array = make_den_curve(
                    zmol[frame], int_z_mol, int_dxdy_mol, nmol,
                    nslice, nz, qm, dim)
                save_hdf5(
                    count_data_file + '_count_corr', count_corr_array,
                    frame, mode_count_corr)
