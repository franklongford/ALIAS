import numpy as np
import tables


def make_earray(file_name, arrays, atom, sizes):
    """
    General purpose algorithm to create an empty earray

    Parameters
    ----------
    file_name:  str
        Name of file
    arrays:  str, list
        List of references for arrays in data table
    atom:  type
        Type of data in earray
    sizes:  int, tuple
        Shape of arrays in data set
    """

    with tables.open_file(file_name, 'w') as outfile:
        for i, array in enumerate(arrays):
            outfile.create_earray(
                outfile.root, array, atom, sizes[i])


def make_hdf5(file_path, shape, datatype):
    """
    General purpose algorithm to create an empty hdf5 file

    Parameters
    ----------
    file_path:  str
        Path name of hdf5 file
    shape:  int, tuple
        Shape of dataset in hdf5 file
    datatype:  type
        Data type of dataset
    """

    shape = (0,) + shape

    make_earray(
        file_path + '.hdf5', ['dataset'],
        datatype,
        [shape]
    )


def load_hdf5(file_path, frame='all'):
    """
    General purpose algorithm to load an array from a hdf5 file

    Parameters
    ----------
    file_path:  str
        Path name of hdf5 file
    frame:  int (optional)
        Trajectory frame to load

    Returns
    -------
    array:  array_like (float);
        Data array to be loaded, same shape as object
        'dataset' in hdf5 file
    """

    with tables.open_file(file_path + '.hdf5', 'r') as infile:
        if frame == 'all':
            array = infile.root.dataset[:]
        else:
            array = infile.root.dataset[frame]

    return array


def save_hdf5(file_path, array, frame, mode='a'):
    """
    General purpose algorithm to save an array from a single
    frame a hdf5 file

    Parameters
    ----------
    file_path:  str
        Path name of hdf5 file
    array:  array_like (float);
        Data array to be saved, must be same shape as object
        'dataset' in hdf5 file
    frame:  int
        Trajectory frame to save
    mode:  str (optional)
        Option to append 'a' to hdf5 file or overwrite 'r+'
        existing data
    """

    if not mode:
        return

    shape = (1,) + array.shape

    with tables.open_file(file_path + '.hdf5', mode) as outfile:
        assert outfile.root.dataset.shape[1:] == shape[1:]

        if mode.lower() == 'a':
            write_array = np.zeros(shape)
            write_array[0] = array
            outfile.root.dataset.append(write_array)

        elif mode.lower() == 'r+':
            outfile.root.dataset[frame] = array


def _hdf5_shape(file_path):
    """
    General purpose algorithm to load the dataset shape
    in a hdf5 file

    Parameters
    ----------
    file_path:  str
        Path name of hdf5 file

    Returns
    -------
    shape_hdf5:  int, tuple
        Shape of object dataset in hdf5 file
    """
    with tables.open_file(file_path + '.hdf5', 'r') as infile:
        shape_hdf5 = infile.root.dataset.shape

    return shape_hdf5


def shape_check_hdf5(file_path, shape):
    """
    General purpose algorithm to check the shape the dataset
    in a hdf5 file

    Parameters
    ----------
    file_path:  str
        Path name of hdf5 file
    shape: tuple

    Returns
    -------
    Bool:  int, tuple
        Whether shape is the same as hdf5 file
    """
    return _hdf5_shape(file_path) == shape


def frame_check_hdf5(file_path, nframe):
    """
    General purpose algorithm to check the shape the dataset
    in a hdf5 file

    Parameters
    ----------
    file_path:  str
        Path name of hdf5 file
    nframe: int
        Number of frames to check in hd5f dataset

    Returns
    -------
    Bool:  int, tuple
        Whether numeber of frames is the same as hdf5 file
    """
    return _hdf5_shape(file_path)[0] <= nframe


def mode_check_hdf5(frame_check, ow=False):
    if frame_check:
        return 'a'
    elif ow:
        return 'r+'
    return False
