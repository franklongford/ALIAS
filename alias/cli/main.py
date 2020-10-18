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

from alias.io.command_line_input import enter_file
from alias.src.run_alias import run_alias
from alias.src.alias_options import AliasOptions
from alias.src.utilities import print_alias
from alias.version import __version__


@click.command()
@click.version_option(version=__version__)
@click.option(
    '--topology', type=click.Path(exists=True), default=None,
    help='File path of optional topology file for system'
)
@click.option(
    '--checkpoint', is_flag=True, default=None,
    help='Provide checkpoint file with intrinsic surface parameters'
)
@click.option(
    '--debug', is_flag=True, default=False,
    help="Prints extra debug information in pyfibre.log"
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
@click.argument(
    'trajectory', type=click.Path(exists=True),
    required=True, default=None
)
def alias(trajectory, topology, debug, checkpoint,
          ow_coeff, ow_recon, ow_pos, ow_intpos, ow_hist,
          ow_dist):

    # Initialising log
    if debug:
        logging.basicConfig(filename="pyfibre.log", filemode="w",
                            level=logging.DEBUG)
    else:
        logging.basicConfig(filename="pyfibre.log", filemode="w",
                            level=logging.INFO)

    log = logging.getLogger(__name__)
    log.info(f'Starting ALIAS version {__version__}')
    print_alias()

    # Get trajectory file path and topology is required
    trajectory = enter_file('Trajectory', file_path=trajectory)
    log.info(f'Using trajectory file {trajectory}')

    if topology is not None:
        topology = enter_file('Topology', file_path=topology)
        log.info(f'Using topology file {topology}')

    # Collate options for overwriting files
    options = AliasOptions(
        ow_coeff, ow_recon, ow_pos,
        ow_intpos, ow_hist, ow_dist
    )

    run_alias(
        trajectory, options,
        checkpoint=checkpoint, topology=topology)
