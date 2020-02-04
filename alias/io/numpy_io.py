import numpy as np


def load_npy(file_path, frames=[]):
    """
    load_npy(file_path, frames=[])

    General purpose algorithm to load an array from a npy file

    Parameters
    ----------

    file_path:  str
        Path name of npy file
    frames:  int, list (optional)
        Trajectory frames to load

    Returns
    -------

    array:  array_like (float);
        Data array to be loaded
    """

    if len(frames) == 0:
        array = np.load(file_path + '.npy', mmap_mode='r')
    else:
        array = np.load(file_path + '.npy', mmap_mode='r')[frames]

    return array
