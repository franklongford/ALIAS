"""
*************** UTILITIES MODULE *******************

Generic functions to be used across multiple programs
within ALIAS.

***********************************************************
Created 24/11/2016 by Frank Longford

Contributors: Frank Longford

Last modified 27/2/2018 by Frank Longford
"""
import os

import numpy as np
import mdtraj as md

from alias.version import __version__

SQRT2 = np.sqrt(2.)
SQRTPI = np.sqrt(np.pi)


def numpy_remove(list1, list2):
    """
    Deletes overlapping elements of list2 from list1
    """

    return np.delete(list1, np.where(np.isin(list1, list2)))


def load_traj_frame(traj_file, top_file=None):
    """
    Returns single frame of input trajectory and topology file using mdtraj

    Parameters
    ----------
    traj_file:  str
            Trajectory file name
    top_file:  str, optional
            Topology file name

    Returns
    -------
    traj:  mdtraj obj
            Single frame of mdtraj trajectory object
    """

    traj = md.load_frame(traj_file, 0, top=top_file)

    return traj


def bubble_sort(array, key):
    """
    bubble_sort(array, key)

    Sorts array and key by order of elements of key
    """

    for passnum in range(len(array)-1, 0, -1):
        for i in range(passnum):
            if key[i] > key[i+1]:
                temp = array[i]
                array[i] = array[i+1]
                array[i+1] = temp

                temp = key[i]
                key[i] = key[i+1]
                key[i+1] = temp


def unit_vector(vector, axis=-1):
    """
    unit_vector(vector, axis=-1)

    Returns unit vector of vector
    """

    vector = np.array(vector)
    magnitude_2 = np.sum(vector.T**2, axis=axis)
    u_vector = np.sqrt(vector**2 / magnitude_2) * np.sign(vector)

    return u_vector


def print_alias():

    logo = ""

    logo += ' ' + '_' * 43
    logo += r"|                   __ __             ____  |\n"
    logo += r"|     /\     |        |       /\     /      |\n"
    logo += r"|    /  \    |        |      /  \    \___   |\n"
    logo += r"|   /___ \   |        |     /___ \       \  |\n"
    logo += r"|  /      \  |____  __|__  /      \  ____/  |\n"
    logo += "|" + '_' * 43 + '|' + f"  v{__version__}\n"
    logo += "\n    Air-Liquid Interface Analysis Suite \n"

    print(logo)


def create_surface_file_path(file_name, directory, q_m, n0,
                             phi, n_frame, recon):

    coeff_ext = f'_{q_m}_{n0}_{int(1. / phi + 0.5)}_{n_frame}'

    if recon:
        coeff_ext += '_r'

    file_path = os.path.join(
        directory, file_name + coeff_ext)

    return file_path
