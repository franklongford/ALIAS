import os
import logging

import numpy as np

from alias.io.numpy_io import load_npy
from alias.io.checkfile_io import (
    save_checkfile
)
from alias.src.positions import batch_coordinate_loader
from alias.src.intrinsic_sampling_method import create_intrinsic_surfaces
from alias.src.intrinsic_analysis import (
    create_intrinsic_positions_dxdyz,
    create_intrinsic_den_curve_hist,
    av_intrinsic_distributions
)

log = logging.getLogger(__name__)


def run_alias(trajectory, alias_options, surface_parameters, topology=None):
    """Peform ALIAS on given trajectory, including options """

    traj_dir = os.path.dirname(trajectory)

    alias_dir = os.path.join(traj_dir, 'alias_analysis')
    data_dir = os.path.join(alias_dir, 'data')
    figure_dir = os.path.join(alias_dir, 'figures')

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    if not os.path.exists(figure_dir):
        os.mkdir(figure_dir)

    file_name, _ = os.path.splitext(trajectory)
    file_name = os.path.basename(file_name)

    log.info("Loading trajectory file {} using {} topology".format(
        trajectory, topology))
    checkfile_name = os.path.join(alias_dir, file_name + '_chk.json')

    surface_parameters.load_traj_parameters(
        trajectory, topology)

    surface_parameters.select_residue()
    checkfile = surface_parameters.serialize()
    save_checkfile(checkfile, checkfile_name)

    surface_parameters.select_masses()
    checkfile = surface_parameters.serialize()
    save_checkfile(checkfile, checkfile_name)

    surface_parameters.select_center_of_mass()
    checkfile = surface_parameters.serialize()
    save_checkfile(checkfile, checkfile_name)

    com_ref = '_'.join([
        surface_parameters.com_mode,
        '-'.join([str(m) for m in surface_parameters.com_sites])])
    file_name = f"{file_name}_{com_ref}"

    surface_parameters.select_orientation_vector()
    checkfile = surface_parameters.serialize()
    save_checkfile(checkfile, checkfile_name)

    pos_dir = os.path.join(data_dir, 'pos')
    if not os.path.exists(pos_dir):
        os.mkdir(pos_dir)
    pos_file_name = os.path.join(pos_dir, file_name)

    try:
        mol_traj = load_npy(pos_file_name + f'{surface_parameters.n_frames}_mol_traj')
        com_traj = load_npy(pos_file_name + f'{surface_parameters.n_frames}_com_traj')
        mol_vec = load_npy(pos_file_name + f'{surface_parameters.n_frames}_mol_vec')
        cell_dim = load_npy(pos_file_name + f'{surface_parameters.n_frames}_cell_dim')
    except (FileNotFoundError, IOError):

        mol_traj, com_traj, cell_dim, mol_vec = batch_coordinate_loader(
            trajectory, surface_parameters, topology=topology
        )

        np.save(pos_file_name + f'{surface_parameters.n_frames}_mol_traj', mol_traj)
        np.save(pos_file_name + f'{surface_parameters.n_frames}_mol_vec', mol_vec)
        np.save(pos_file_name + f'{surface_parameters.n_frames}_com_traj', com_traj)
        np.save(pos_file_name + f'{surface_parameters.n_frames}_cell_dim', cell_dim)

    surface_parameters.select_mol_sigma()
    checkfile = surface_parameters.serialize()
    save_checkfile(checkfile, checkfile_name)

    surface_parameters.n_frames = mol_traj.shape[0]
    mean_cell_dim = np.mean(cell_dim, axis=0)
    surface_parameters.cell_dim = mean_cell_dim.tolist()

    checkfile = surface_parameters.serialize()
    save_checkfile(checkfile, checkfile_name)

    print(f"Simulation cell xyz dimensions in Angstoms: "
          f"{surface_parameters.area}\n")

    print("\n------STARTING INTRINSIC SAMPLING-------\n")
    print("Max wavelength = {:12.4f} sigma   Min wavelength = {:12.4f} sigma".format(
        surface_parameters.q_max, surface_parameters.q_min))
    print("Max frequency qm = {:6d}".format(
        surface_parameters.q_m))

    surface_parameters.select_pivot_density(file_name, data_dir)
    checkfile = surface_parameters.serialize()
    save_checkfile(checkfile, checkfile_name)

    freq_range = range(1, surface_parameters.q_m+1)
    print("\nResolution parameters:")
    print("\n{:12s} | {:12s} | {:12s}".format('qu', "lambda (sigma)", "lambda (nm)"))
    print("-" * 14 * 5)

    for q_u in freq_range:
        print("{:12d} | {:12.4f} | {:12.4f}".format(
            q_u,
            surface_parameters.wavelength(q_u),
            surface_parameters.wavelength(q_u) * surface_parameters.mol_sigma / 10))
    print("")

    create_intrinsic_surfaces(
        data_dir, file_name, mean_cell_dim, surface_parameters.q_m,
        surface_parameters.n_pivots, surface_parameters.phi,
        surface_parameters.mol_sigma, surface_parameters.n_frames,
        recon=surface_parameters.recon, ncube=surface_parameters.n_cube,
        vlim=surface_parameters.v_lim, tau=surface_parameters.tau,
        max_r=surface_parameters.max_r,
        ow_coeff=alias_options.ow_coeff, ow_recon=alias_options.ow_recon)

    create_intrinsic_positions_dxdyz(
        data_dir, file_name, surface_parameters.n_mol,
        surface_parameters.n_frames, surface_parameters.q_m,
        surface_parameters.n_pivots, surface_parameters.phi,
        mean_cell_dim,
        recon=surface_parameters.recon,
        ow_pos=alias_options.ow_intpos)

    create_intrinsic_den_curve_hist(
        data_dir, file_name, surface_parameters.q_m, surface_parameters.n_pivots,
        surface_parameters.phi, surface_parameters.n_frames,
        surface_parameters, surface_parameters.n_slice,
        surface_parameters.cell_dim,
        recon=surface_parameters.recon,
        ow_hist=alias_options.ow_hist)

    av_intrinsic_distributions(
        data_dir, file_name, surface_parameters.cell_dim,
        surface_parameters.n_slice, surface_parameters.q_m,
        surface_parameters.n_pivots, surface_parameters.phi,
        surface_parameters.n_frames, surface_parameters.n_frames,
        recon=surface_parameters.recon,
        ow_dist=alias_options.ow_dist)

    print("\n---- ENDING PROGRAM ----\n")
