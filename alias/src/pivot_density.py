import os
import sys

import numpy as np
import tables

from alias.io.hdf5_io import (
    make_hdf5,
    shape_check_hdf5,
    load_hdf5,
    save_hdf5
)
from alias.io.numpy_io import load_npy
from alias.src.intrinsic_sampling_method import build_surface
from alias.src.utilities import create_surface_file_path


def pivot_diffusion(file_name, surface_dir, mol_traj, cell_dim, mol_vec,
                    surf_param, n_frame=20):

    print("Density Coefficient = {}".format(surf_param.pivot_density))
    print("Using pivot number = {}".format(surf_param.n_pivots))

    tot_piv_n1 = np.zeros((n_frame, surf_param.n_pivots), dtype=int)
    tot_piv_n2 = np.zeros((n_frame, surf_param.n_pivots), dtype=int)

    coeff_file_name = create_surface_file_path(
        file_name, surface_dir, surf_param.q_m,
        surf_param.n_pivots, surf_param.phi,
        surf_param.n_frames, surf_param.recon
    )

    if not os.path.exists(coeff_file_name + '_coeff.hdf5'):
        make_hdf5(coeff_file_name + '_coeff',
                  (2, surf_param.n_waves ** 2), tables.Float64Atom())
        make_hdf5(coeff_file_name + '_pivot',
                  (2, surf_param.n_pivots), tables.Int64Atom())

    for frame in range(n_frame):
        dim = cell_dim[frame]

        "Checking number of frames in coeff and pivot files"
        frame_check_coeff = (shape_check_hdf5(coeff_file_name + '_coeff')[0] <= frame)
        frame_check_pivot = (shape_check_hdf5(coeff_file_name + '_pivot')[0] <= frame)

        if frame_check_coeff:
            mode_coeff = 'a'
        else:
            mode_coeff = False

        if frame_check_pivot:
            mode_pivot = 'a'
        else:
            mode_pivot = False

        if not mode_coeff and not mode_pivot:
            pivot = load_hdf5(coeff_file_name + '_pivot', frame)
        else:
            sys.stdout.write(f"Optimising Intrinsic Surface coefficients: frame {frame}\n")
            sys.stdout.flush()

            if frame == 0:
                surf_0 = [-dim[2] / 4, dim[2] / 4]
            else:
                index = (2 * surf_param.q_m + 1) ** 2 / 2
                coeff = load_hdf5(coeff_file_name + '_coeff', frame - 1)
                surf_0 = [coeff[0][index], coeff[1][index]]

            coeff, pivot = build_surface(mol_traj[frame, :, 0], mol_traj[frame, :, 1], mol_traj[frame, :, 2], dim,
                                         surf_param.q_m, surf_param.n_pivots, surf_param.phi, surf_param.tau,
                                         surf_param.max_r, ncube=surf_param.n_cube, vlim=surf_param.v_lim,
                                         recon=surf_param.recon, surf_0=surf_0, zvec=mol_vec[frame, :, 2])

            save_hdf5(coeff_file_name + '_coeff', coeff, frame, mode_coeff)
            save_hdf5(coeff_file_name + '_pivot', pivot, frame, mode_pivot)

        tot_piv_n1[frame] += pivot[0]
        tot_piv_n2[frame] += pivot[1]

    ex_1, ex_2 = mol_exchange(tot_piv_n1, tot_piv_n2)

    return ex_1, ex_2


def optimise_pivot_diffusion(file_name, directory, surf_param, mol_vec=None, start_density=0.85, step_density=0.05,
                             n_frame=20, precision=5E-4, gamma=0.5):
    """
    Routine to find optimised pivot density coefficient ns and pivot number n0 based on lowest pivot diffusion rate

    Parameters
    ----------
    file_name:  str
        File name of trajectory being analysed.
    directory:  str
        File path of directory of alias analysis.
    mol_traj:  array_like
        Trajectory of molecular coordinates
    com_traj:  array_like
        Trajectory of molecular centre of mass
    cell_dim:  array_like
        Trajectory of simulation cell dimensions
    surf_param: instance SurfaceParameters
        Parameters for intrinsic surface builder
    mol_vec: array_like
        Trajectory of molecular orientations
    start_density:  float
        Initial value for pivot density optimisation routine
    step_density:  float
        Search step difference between each pivot density value ns
    n_frame:  (optional) int
        Number of trajectory frames to perform optimisation with
    precision:  (optional) float
        Precision for optimisation process
    gamma:  (optional) float
        Step length coefficient for optimisation scheme
    """

    pos_dir = os.path.join(directory, 'pos')
    surface_dir = os.path.join(directory, 'surface')

    if surf_param.n_frames < n_frame:
        n_frame = surf_param.n_frames

    if not os.path.exists(surface_dir):
        os.mkdir(surface_dir)

    pos_file_name = os.path.join(pos_dir, file_name)
    mol_traj = load_npy(pos_file_name + '_mol_traj', frames=range(n_frame))
    com_traj = load_npy(pos_file_name + '_com_traj', frames=range(n_frame))
    cell_dim = load_npy(pos_file_name + '_cell_dim', frames=range(n_frame))

    mol_ex_1 = []
    mol_ex_2 = []
    density_array = []
    derivative = []

    com_tile = np.moveaxis(
        np.tile(com_traj, (surf_param.n_mols, 1, 1)),
        [0, 1, 2], [2, 1, 0])[2]
    mol_traj[:, :, 2] -= com_tile

    surf_param.pivot_density = start_density
    optimising = True

    print("Surface pivot density precision = {}".format(precision))

    while optimising:

        density_array.append(surf_param.pivot_density)

        ex_1, ex_2 = pivot_diffusion(
            file_name, surface_dir, mol_traj, cell_dim, mol_vec,
            surf_param, n_frame=n_frame
        )

        mol_ex_1.append(ex_1)
        mol_ex_2.append(ex_2)

        av_mol_ex = (np.array(mol_ex_1) + np.array(mol_ex_2)) / 2.
        print("Average Pivot Diffusion Rate = {} mol / frame".format(av_mol_ex[-1]))

        if len(av_mol_ex) > 1:
            step_size = (av_mol_ex[-1] - av_mol_ex[-2])
            derivative.append(step_size / (density_array[-1] - density_array[-2]))

            check = abs(step_size) <= precision
            if check:
                optimising = False
            else:
                # if len(derivative) > 1: gamma = (NS[-1] - NS[-2]) / (derivative[-1] - derivative[-2])
                surf_param.pivot_density -= gamma * derivative[-1]
                print("Optimal pivot density not found.\nSurface density coefficient step size = |{}| > {}\n".format(
                    step_size, precision))

        else:
            surf_param.pivot_density += step_density
            print("Optimal pivot density not found.\nSurface density coefficient step size = |{}| > {}\n".format(
                step_density / gamma, precision))

    surf_param.pivot_density = density_array[np.argmin(av_mol_ex)]

    print("Optimal pivot density found = {}\nSurface density coefficient step size = |{}| < {}\n".format(
        surf_param.pivot_density, step_size, precision))
    print("Optimal number of pivots = {}".format(surf_param.n_pivots))

    remove_unwanted_files(file_name, surface_dir, density_array, surf_param)


def remove_unwanted_files(file_name, surface_dir, density_array, surf_param):

    for density in density_array:

        if density != surf_param.pivot_density:

            n_pivots = int(surf_param.area * density / surf_param.mol_sigma ** 2)

            coeff_file_name = create_surface_file_path(
                file_name, surface_dir, surf_param.q_m,
                n_pivots, surf_param.phi,
                surf_param.n_frames, surf_param.recon
            )

            os.remove(coeff_file_name + '_coeff.hdf5')
            os.remove(coeff_file_name + '_pivot.hdf5')


def mol_exchange(piv_1, piv_2):
    """
    mol_exchange(piv_1, piv_2, nframe, n0)

    Calculates average diffusion rate of surface pivot molecules between frames

    Parameters
    ----------

    piv_1:  float, array_like; shape=(nframe, n0)
        Molecular pivot indices of upper surface at each frame
    piv_2:  float, array_like; shape=(nframe, n0)
        Molecular pivot indices of lower surface at each frame

    Returns
    -------

    diff_rate1: float
        Diffusion rate of pivot molecules in mol frame^-1 of upper surface
    diff_rate2: float
        Diffusion rate of pivot molecules in mol frame^-1 of lower surface

    """

    nframe, n0 = piv_1.shape

    n_1 = 0
    n_2 = 0

    for frame in range(nframe-1):

        n_1 += len(set(piv_1[frame]) - set(piv_1[frame+1]))
        n_2 += len(set(piv_2[frame]) - set(piv_2[frame+1]))

    diff_rate1 = n_1 / (n0 * float(nframe-1))
    diff_rate2 = n_2 / (n0 * float(nframe-1))

    return diff_rate1, diff_rate2
