import os


def make_directory(directory):

    if not os.path.exists(directory):
        os.mkdir(directory)
