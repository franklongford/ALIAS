import json


def make_checkfile(checkfile_name):
    """
    Creates checkfile for analysis, storing key paramtere
    """

    checkfile = {}

    save_checkfile(checkfile, checkfile_name)

    return checkfile


def save_checkfile(checkfile, checkfile_name):
    """
    Saves checkfile, storing key paramtere
    """

    with open(checkfile_name, 'w') as outfile:
        json.dump(checkfile, outfile, indent=4)


def load_checkfile(checkfile_name):
    """
    Reads checkfile to lookup stored key paramters
    """

    with open(checkfile_name, 'r') as infile:
        checkfile = json.load(infile)

    return checkfile


def update_checkfile(checkfile_name, symb, obj):
    """
    Updates checkfile parameter

    Parameters
    ----------
    checkfile_name:  str
            Checkfile path + name
    symb:  str
            Key for checkfile dictionary of object obj
    obj:
            Parameter to be saved

    Returns
    -------
    checkfile:  dict
            Dictionary of key parameters
    """

    checkfile = load_checkfile(checkfile_name)

    checkfile[symb] = obj

    save_checkfile(checkfile, checkfile_name)

    return checkfile
