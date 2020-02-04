"""
*************** UTILITIES MODULE *******************

Generic functions to be used across multiple programs
within ALIAS.

***********************************************************
Created 24/11/2016 by Frank Longford

Contributors: Frank Longford

Last modified 27/2/2018 by Frank Longford
"""

import numpy as np

from scipy.signal import convolve
import mdtraj as md

from alias.version import __version__

SQRT2 = np.sqrt(2.)
SQRTPI = np.sqrt(np.pi)


def numpy_remove(list1, list2):
    """
    Deletes overlapping elements of list2 from list1
    """

    return np.delete(list1, np.where(np.isin(list1, list2)))


def get_sim_param(traj_file, top_file):
    """
    Returns selected parameters of input trajectory and topology file using mdtraj

    Parameters
    ----------
    top_dir:  str
            Directory of topology file
    traj_dir:  str
            Directory of trajectory file
    traj_file:  str
            Trajectory file name
    top_file:  str
            Topology file name

    Returns
    -------
    traj:  mdtraj obj
            Mdtraj trajectory object
    mol:  str, list
            List of residue types in simulation cell
    nframe:  int
            Number of frames sampled in traj_file
    dim:  float, array_like; shape=(3):
            Simulation cell dimensions (angstroms)
    """

    traj = md.load_frame(traj_file, 0, top=top_file)
    mol = list(set([molecule.name for molecule in traj.topology.residues]))

    return traj, mol


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


def linear(x, m, c):
    return m * x + c


def gaussian(x, mean, std):
    return np.exp(-(x-mean)**2 / (2 * std**2)) / (SQRT2 * std * SQRTPI)


def gaussian_convolution(array, centre, delta, dim, nslice):
    """
    Convolution of distributions 'arrays' using a normal probability distribution with mean=centres and variance=deltas

    Parameters
    ----------
    array:  float, array_like; shape=(nslice)
        Array to convolute
    centre: float
        Mean value for normal probability distribution
    delta: float
        Variance for normal probability distribution
    dim:  float, array_like; shape=(3)
        XYZ dimensions of simulation cell
    nslice: int
        Number of bins in density histogram along axis normal to surface

    Returns
    -------
    conv_array:  float, array_like; shape=(nslice)
        Convoluted array
    """

    std = np.sqrt(delta)
    lslice = dim[2] / nslice
    length = int(std / lslice) * 10
    ZG = np.arange(0, dim[2], lslice)

    index = nslice / 8
    array = np.roll(array, -index)

    gaussian_array = gaussian(ZG, centre+ZG[index], std) * lslice
    conv_array = convolve(array, gaussian_array, mode='same')

    """
    import matplotlib.pyplot as plt
    plt.figure(100)
    plt.plot(ZG, gaussian_array)
    plt.plot(ZG, array)
    plt.plot(ZG, conv_array)
    plt.show()
    #"""

    return conv_array


def print_alias():

    print(' '+ '_' * 43)
    print("|                   __ __             ____  |")
    print("|     /\     |        |       /\     /      |")
    print("|    /  \    |        |      /  \    \___   |")
    print("|   /___ \   |        |     /___ \       \  |")
    print("|  /      \  |____  __|__  /      \  ____/  |")
    print(f"|'+ '_' * 43 + '|' + '  v{__version__}")
    print("\n    Air-Liquid Interface Analysis Suite \n")
