import os
import sys

import numpy as np
import tables

from alias.io.hdf5_io import make_hdf5, shape_check_hdf5, load_hdf5, save_hdf5
from alias.io.numpy_io import load_npy
from alias.src.intrinsic_sampling_method import build_surface


def optimise_ns_diff(directory, file_name, nmol, nframe, qm, phi, dim, mol_sigma, start_ns, step_ns=0.05, recon=False,
                     nframe_ns = 20, ncube=3, vlim=3, tau=0.5, max_r=1.5, precision=5E-4, gamma=0.5):
    """
    optimise_ns(directory, file_name, nmol, nframe, qm, phi, ncube, dim, mol_sigma, start_ns, step_ns=0.05, nframe_ns=20, vlim=3)

    Routine to find optimised pivot density coefficient ns and pivot number n0 based on lowest pivot diffusion rate

    Parameters
    ----------

    directory:  str
        File path of directory of alias analysis.
    file_name:  str
        File name of trajectory being analysed.
    nmol:  int
        Number of molecules in simulation
    nframe: int
        Number of trajectory frames in simulation
    qm:  int
        Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
    phi:  float
        Weighting factor of minimum surface area term in surface optimisation function
    dim:  float, array_like; shape=(3)
        XYZ dimensions of simulation cell
    mol_sigma:  float
        Radius of spherical molecular interaction sphere
    start_ns:  float
        Initial value for pivot density optimisation routine
    step_ns:  float
        Search step difference between each pivot density value ns
    nframe_ns:  (optional) int
        Number of trajectory frames to perform optimisation with

    Returns
    -------

    opt_ns: float
        Optimised surface pivot density parameter
    opt_n0: int
        Optimised number of pivot molecules

    """

    pos_dir = directory + 'pos/'
    surf_dir = directory + 'surface/'

    if not os.path.exists(surf_dir): os.mkdir(surf_dir)

    mol_ex_1 = []
    mol_ex_2 = []
    NS = []
    derivative = []

    n_waves = 2 * qm + 1
    max_r *= mol_sigma
    tau *= mol_sigma

    xmol = load_npy(pos_dir + file_name + '_{}_xmol'.format(nframe), frames=range(nframe_ns))
    ymol = load_npy(pos_dir + file_name + '_{}_ymol'.format(nframe), frames=range(nframe_ns))
    zmol = load_npy(pos_dir + file_name + '_{}_zmol'.format(nframe), frames=range(nframe_ns))
    COM = load_npy(pos_dir + file_name + '_{}_com'.format(nframe), frames=range(nframe_ns))
    zvec = load_npy(pos_dir + file_name + '_{}_zvec'.format(nframe), frames=range(nframe_ns))

    com_tile = np.moveaxis(np.tile(COM, (nmol, 1, 1)), [0, 1, 2], [2, 1, 0])[2]
    zmol = zmol - com_tile

    if nframe < nframe_ns: nframe_ns = nframe
    ns = start_ns
    optimising = True

    print("Surface pivot density precision = {}".format(precision))

    while optimising:

        NS.append(ns)
        n0 = int(dim[0] * dim[1] * ns / mol_sigma**2)

        print("Density Coefficient = {}".format(ns))
        print("Using pivot number = {}".format(n0))

        tot_piv_n1 = np.zeros((nframe_ns, n0), dtype=int)
        tot_piv_n2 = np.zeros((nframe_ns, n0), dtype=int)

        file_name_coeff = '{}_{}_{}_{}_{}'.format(file_name, qm, n0, int(1./phi + 0.5), nframe)

        if recon: file_name_coeff += '_r'

        if not os.path.exists('{}/surface/{}_coeff.hdf5'.format(directory, file_name_coeff)):
            make_hdf5(surf_dir + file_name_coeff + '_coeff', (2, n_waves**2), tables.Float64Atom())
            make_hdf5(surf_dir + file_name_coeff + '_pivot', (2, n0), tables.Int64Atom())

        for frame in range(nframe_ns):
            "Checking number of frames in coeff and pivot files"
            frame_check_coeff = (shape_check_hdf5(surf_dir + file_name_coeff + '_coeff')[0] <= frame)
            frame_check_pivot = (shape_check_hdf5(surf_dir + file_name_coeff + '_pivot')[0] <= frame)

            if frame_check_coeff: mode_coeff = 'a'
            else: mode_coeff = False

            if frame_check_pivot: mode_pivot = 'a'
            else: mode_pivot = False

            if not mode_coeff and not mode_pivot:
                pivot = load_hdf5(surf_dir + file_name_coeff + '_pivot', frame)
            else:
                sys.stdout.write("Optimising Intrinsic Surface coefficients: frame {}\n".format(frame))
                sys.stdout.flush()

                if frame == 0: surf_0 = [-dim[2]/4, dim[2]/4]
                else:
                    index = (2 * qm + 1)**2 / 2
                    coeff = load_hdf5(surf_dir + file_name_coeff + '_coeff', frame-1)
                    surf_0 = [coeff[0][index], coeff[1][index]]

                coeff, pivot = build_surface(xmol[frame], ymol[frame], zmol[frame], dim, mol_sigma,
                                             qm, n0, phi, tau, max_r, ncube=ncube, vlim=vlim, surf_0=surf_0,
                                             recon=recon, zvec=zvec[frame])

                save_hdf5(surf_dir + file_name_coeff + '_coeff', coeff, frame, mode_coeff)
                save_hdf5(surf_dir + file_name_coeff + '_pivot', pivot, frame, mode_pivot)

            tot_piv_n1[frame] += pivot[0]
            tot_piv_n2[frame] += pivot[1]

        ex_1, ex_2 = mol_exchange(tot_piv_n1, tot_piv_n2, nframe_ns, n0)

        mol_ex_1.append(ex_1)
        mol_ex_2.append(ex_2)

        av_mol_ex = (np.array(mol_ex_1) + np.array(mol_ex_2)) / 2.
        print("Average Pivot Diffusion Rate = {} mol / frame".format(av_mol_ex[-1]))

        if len(av_mol_ex) > 1:
            step_size = (av_mol_ex[-1] - av_mol_ex[-2])
            derivative.append(step_size / (NS[-1] - NS[-2]))
            #min_arg = np.argsort(abs(NS[-1] - np.array(NS)))
            #min_derivative = (av_mol_ex[min_arg[0]] - av_mol_ex[min_arg[1]]) / (NS[min_arg[0]] - NS[min_arg[1]])
            #derivative.append(min_derivative)

            check = abs(step_size) <= precision
            if check: optimising = False
            else:
                #if len(derivative) > 1: gamma = (NS[-1] - NS[-2]) / (derivative[-1] - derivative[-2])
                ns -= gamma * derivative[-1]
                print("Optimal pivot density not found.\nSurface density coefficient step size = |{}| > {}\n".format(step_size, precision))

        else:
            ns += step_ns
            print("Optimal pivot density not found.\nSurface density coefficient step size = |{}| > {}\n".format(step_ns / gamma, precision))


    opt_ns = NS[np.argmin(av_mol_ex)]
    opt_n0 = int(dim[0] * dim[1] * opt_ns / mol_sigma**2)

    print("Optimal pivot density found = {}\nSurface density coefficient step size = |{}| < {}\n".format(opt_ns, step_size, precision))
    print("Optimal number of pivots = {}".format(opt_n0))

    for ns in NS:
        if ns != opt_ns:
            n0 = int(dim[0] * dim[1] * ns / mol_sigma**2)
            file_name_coeff = '{}_{}_{}_{}_{}'.format(file_name, qm, n0, int(1./phi + 0.5), nframe)
            if recon: file_name_coeff += '_r'
            os.remove(surf_dir + file_name_coeff + '_coeff.hdf5')
            os.remove(surf_dir + file_name_coeff + '_pivot.hdf5')

    return opt_ns, opt_n0


def mol_exchange(piv_1, piv_2, nframe, n0):
    """
    mol_exchange(piv_1, piv_2, nframe, n0)

    Calculates average diffusion rate of surface pivot molecules between frames

    Parameters
    ----------

    piv_1:  float, array_like; shape=(nframe, n0)
        Molecular pivot indicies of upper surface at each frame
    piv_2:  float, array_like; shape=(nframe, n0)
        Molecular pivot indicies of lower surface at each frame
    nframe:  int
        Number of frames to sample over
    n0:  int
        Number of pivot molecules in each surface

    Returns
    -------

    diff_rate1: float
        Diffusion rate of pivot molecules in mol frame^-1 of upper surface
    diff_rate2: float
        Diffusion rate of pivot molecules in mol frame^-1 of lower surface

    """
    n_1 = 0
    n_2 = 0

    for frame in range(nframe-1):

        n_1 += len(set(piv_1[frame]) - set(piv_1[frame+1]))
        n_2 += len(set(piv_2[frame]) - set(piv_2[frame+1]))

    diff_rate1 = n_1 / (n0 * float(nframe-1))
    diff_rate2 = n_2 / (n0 * float(nframe-1))

    return diff_rate1, diff_rate2