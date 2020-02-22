"""
*************** MAIN INTERFACE MODULE *******************

Main program, ALIAS: Air-Liquid Interface Analysis Suite

***********************************************************
Created 24/11/2016 by Frank Longford

Contributors: Frank Longford

Last modified 27/2/2018 by Frank Longford

"""

import click
import logging
import os

from alias.src.run_alias import run_alias
from alias.src.alias_options import AliasOptions
from alias.src.surface_parameters import SurfaceParameters
from alias.src.utilities import print_alias
from alias.version import __version__


@click.command()
@click.version_option(version=__version__)
@click.option(
    '--topology', type=click.Path(exists=True), default=None,
    help='File path of optional topology file for system'
)
@click.option(
    '--debug', is_flag=True, default=False,
    help="Prints extra debug information in pyfibre.log"
)
@click.option(
    '--recon', is_flag=True, default=False,
    help='Toggles surface reconstruction routine'
)
@click.option(
    '--ow_coeff', is_flag=True, default=False,
    help='Toggles overwrite of intrinsic surface coefficients'
)
@click.option(
    '--ow_recon', is_flag=True, default=False,
    help='Toggles overwrite of reconstructed intrinsic surface coefficients'
)
@click.option(
    '--ow_pos', is_flag=True, default=False,
    help='Toggles overwrite positions'
)
@click.option(
    '--ow_intpos', is_flag=True, default=False,
    help='Toggles overwrite intrinsic positions'
)
@click.option(
    '--ow_hist', is_flag=True, default=False,
    help='Toggles overwrite histograms of position and angles'
)
@click.option(
    '--ow_dist', is_flag=True, default=False,
    help='Toggles overwrite intrinsic probability distributions'
)
@click.option(
    '--parameters', is_flag=True, default=None,
    help='Provide file with intrinsic surface parameters'
)
@click.argument(
    'trajectory', type=click.Path(exists=True),
    required=True, default=None
)
def alias(trajectory, topology, debug, recon, parameters,
          ow_coeff, ow_recon, ow_pos, ow_intpos, ow_hist, ow_dist,
          ):

    if debug:
        logging.basicConfig(filename="pyfibre.log", filemode="w",
                            level=logging.DEBUG)
    else:
        logging.basicConfig(filename="pyfibre.log", filemode="w",
                            level=logging.INFO)

    print_alias()

    log = logging.getLogger(__name__)

    log.info(f'Starting ALIAS version {__version__}')

    while not os.path.exists(trajectory):
        trajectory = input("\nTrajectory file not recognised: Re-enter file path: ")

    if topology is not None:
        while not os.path.exists(topology):
            topology = input("\nTopology file not recognised: Re-enter file path: ")

    if ow_hist:
        ow_dist = True

    options = AliasOptions(
        recon, ow_coeff, ow_recon, ow_pos,
        ow_intpos, ow_hist, ow_dist
    )

    traj_dir = os.path.dirname(trajectory)
    alias_dir = os.path.join(traj_dir, 'alias_analysis')

    if not os.path.exists(alias_dir):
        os.mkdir(alias_dir)

    file_name, _ = os.path.splitext(trajectory)
    file_name = os.path.basename(file_name)

    if parameters is None:
        parameters = os.path.join(alias_dir, file_name + '_chk.json')

    parameters = SurfaceParameters.from_json(parameters)

    run_alias(trajectory, options, parameters, topology=topology)
