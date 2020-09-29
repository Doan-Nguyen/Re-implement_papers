import os
import sys


def check_make_folder(path):
    """

    """
    if not os.path.isdir(path):
        os.mkdir(path)
    return path


# def check_exist_folder(path):
#     """

#     """
#     Exception

