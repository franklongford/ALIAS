import os
import logging

import numpy as np

from alias.io.numpy_io import load_npy
from alias.io.checkfile_io import (
    save_checkfile
)
from alias.src.positions import batch_coordinate_loader
from alias.src.surface_parameters import SurfaceParameters
from alias.src.intrinsic_sampling_method import (
    create_intrinsic_surfaces
)
from alias.src.intrinsic_analysis import (
    create_intrinsic_positions_dxdyz,
    create_intrinsic_den_curve_hist,
    av_intrinsic_distributions
)
from alias.io.utilities import make_directory

log = logging.getLogger(__name__)


def run_alias(trajectory, alias_options, checkpoint=None, topology=None):
    """Peform ALIAS on given trajectory,"""

    # Obtain directory for trajectory and create analysis
    # directories
    traj_dir = os.path.dirname(trajectory)

    alias_dir = os.path.join(traj_dir, 'alias_analysis')
    data_dir = os.path.join(alias_dir, 'data')
    figure_dir = os.path.join(alias_dir, 'figures')

    make_directory(alias_dir)
    make_directory(data_dir)
    make_directory(figure_dir)

    # Parse file name to obtain base path for analysis
    # files
    file_name, _ = os.path.splitext(trajectory)
    file_name = os.path.basename(file_name)

    # Create a checkpoint file to save intrinsic surface
    # parameters
    if checkpoint is None:
        checkpoint = os.path.join(
            alias_dir, file_name + '_chk.json')

    surf_param = SurfaceParameters.from_json(checkpoint)

    log.info("Loading trajectory file {} using {} topology".format(
        trajectory, topology))
    surf_param.load_traj_parameters(
        trajectory, topology)

    surf_param.select_residue()
    checkfile = surf_param.serialize()
    save_checkfile(checkfile, checkpoint)

    surf_param.select_masses()
    checkfile = surf_param.serialize()
    save_checkfile(checkfile, checkpoint)

    surf_param.select_center_of_mass()
    checkfile = surf_param.serialize()
    save_checkfile(checkfile, checkpoint)

    com_ref = '_'.join([
        surf_param.com_mode,
        '-'.join([str(m) for m in surf_param.com_sites])])
    file_name = f"{file_name}_{com_ref}"

    surf_param.select_orientation_vector()
    checkfile = surf_param.serialize()
    save_checkfile(checkfile, checkpoint)

    pos_dir = os.path.join(data_dir, 'pos')
    if not os.path.exists(pos_dir):
        os.mkdir(pos_dir)
    pos_file_name = os.path.join(pos_dir, file_name)

    try:
        mol_traj = load_npy(pos_file_name + f'{surf_param.n_frames}_mol_traj')
        com_traj = load_npy(pos_file_name + f'{surf_param.n_frames}_com_traj')
        mol_vec = load_npy(pos_file_name + f'{surf_param.n_frames}_mol_vec')
        cell_dim = load_npy(pos_file_name + f'{surf_param.n_frames}_cell_dim')
    except (FileNotFoundError, IOError):

        mol_traj, com_traj, cell_dim, mol_vec = batch_coordinate_loader(
            trajectory, surf_param, topology=topology
        )

        np.save(pos_file_name + f'{surf_param.n_frames}_mol_traj', mol_traj)
        np.save(pos_file_name + f'{surf_param.n_frames}_mol_vec', mol_vec)
        np.save(pos_file_name + f'{surf_param.n_frames}_com_traj', com_traj)
        np.save(pos_file_name + f'{surf_param.n_frames}_cell_dim', cell_dim)

    surf_param.select_mol_sigma()
    checkfile = surf_param.serialize()
    save_checkfile(checkfile, checkpoint)

    surf_param.n_frames = mol_traj.shape[0]
    mean_cell_dim = np.mean(cell_dim, axis=0)
    surf_param.cell_dim = mean_cell_dim.tolist()

    checkfile = surf_param.serialize()
    save_checkfile(checkfile, checkpoint)

    print(f"Simulation cell xyz dimensions in Angstoms: "
          f"{surf_param.area}\n")

    print("\n------STARTING INTRINSIC SAMPLING-------\n")
    print(
        "Max wavelength = {:12.4f} sigma  "
        "Min wavelength = {:12.4f} sigma".format(
            surf_param.q_max, surf_param.q_min)
    )
    print("Max frequency qm = {:6d}".format(
        surf_param.q_m))

    surf_param.select_pivot_density(file_name, data_dir)
    checkfile = surf_param.serialize()
    save_checkfile(checkfile, checkpoint)

    freq_range = range(1, surf_param.q_m+1)
    print("\nResolution parameters:")
    print("\n{:12s} | {:12s} | {:12s}".format(
        'qu', "lambda (sigma)", "lambda (nm)"))
    print("-" * 14 * 5)

    for q_u in freq_range:
        print("{:12d} | {:12.4f} | {:12.4f}".format(
            q_u,
            surf_param.wavelength(q_u),
            surf_param.wavelength(q_u) * surf_param.mol_sigma / 10))
    print("")

    create_intrinsic_surfaces(
        data_dir, file_name, mean_cell_dim, surf_param.q_m,
        surf_param.n_pivots, surf_param.phi,
        surf_param.mol_sigma, surf_param.n_frames,
        recon=surf_param.recon, ncube=surf_param.n_cube,
        vlim=surf_param.v_lim, tau=surf_param.tau,
        max_r=surf_param.max_r,
        ow_coeff=alias_options.ow_coeff,
        ow_recon=alias_options.ow_recon)

    create_intrinsic_positions_dxdyz(
        data_dir, file_name, surf_param.n_mol,
        surf_param.n_frames, surf_param.q_m,
        surf_param.n_pivots, surf_param.phi,
        mean_cell_dim,
        recon=surf_param.recon,
        ow_pos=alias_options.ow_intpos)

    create_intrinsic_den_curve_hist(
        data_dir, file_name, surf_param.q_m, surf_param.n_pivots,
        surf_param.phi, surf_param.n_frames,
        surf_param, surf_param.n_slice,
        surf_param.cell_dim,
        recon=surf_param.recon,
        ow_hist=alias_options.ow_hist)

    av_intrinsic_distributions(
        data_dir, file_name, surf_param.cell_dim,
        surf_param.n_slice, surf_param.q_m,
        surf_param.n_pivots, surf_param.phi,
        surf_param.n_frames, surf_param.n_frames,
        recon=surf_param.recon,
        ow_dist=alias_options.ow_dist)

    print("\n---- ENDING PROGRAM ----\n")
