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
from alias.src.utilities import print_alias
from alias.version import __version__


@click.command()
@click.version_option(version=__version__)
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
    '--ow_intpos', is_flag=True, default=False,
    help='Toggles overwrite network extraction'
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
    'traj_file', type=click.Path(exists=True),
    required=True, default='.'
)
@click.argument(
    'top_file', type=click.Path(exists=True),
    required=True, default='.'
)
def alias(traj_file, top_file, recon,
          ow_coeff, ow_recon, ow_pos, ow_intpos, ow_hist, ow_dist,
          debug):

    if debug:
        logging.basicConfig(filename="pyfibre.log", filemode="w",
                            level=logging.DEBUG)
    else:
        logging.basicConfig(filename="pyfibre.log", filemode="w",
                            level=logging.INFO)

    print_alias()

    while not os.path.exists(traj_file):
        traj_file = input("\nTrajectory file not recognised: Re-enter file path: ")

    while not os.path.exists(top_file):
        top_file = input("\nTopology file not recognised: Re-enter file path: ")

    if ow_hist:
        ow_dist = True

    run_alias(traj_file, top_file, recon, ow_coeff, ow_recon, ow_pos,
              ow_intpos, ow_hist, ow_dist)
