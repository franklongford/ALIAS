import pickle


def make_checkfile(checkfile_name):
    """
    Creates checkfile for analysis, storing key paramtere
    """

    checkfile = {}
    with open(checkfile_name + '.pkl', 'wb') as outfile:
        pickle.dump(checkfile, outfile, pickle.HIGHEST_PROTOCOL)


def read_checkfile(checkfile_name):
    """
    Reads checkfile to lookup stored key paramters
    """

    with open(checkfile_name + '.pkl', 'rb') as infile:
        return pickle.load(infile)


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

    with open(checkfile_name + '.pkl', 'rb') as infile:
        checkfile = pickle.load(infile)
    checkfile[symb] = obj

    with open(checkfile_name + '.pkl', 'wb') as outfile:
        pickle.dump(checkfile, outfile, pickle.HIGHEST_PROTOCOL)
    return checkfile
