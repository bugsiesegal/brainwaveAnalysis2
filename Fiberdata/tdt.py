"""
This file contains a wrapper for the tdt.read_block function.
"""

import tdt
import os


class TDTFile:
    """
    This class is a wrapper for the tdt.read_block function.
    """

    def __init__(self, path):
        """
        This function reads the TDT file and stores the data in the
        class attributes.
        """
        self.filename = os.path.basename(path)
        self.data = tdt.read_block(path)

    def __str__(self):
        """
        This function returns a string representation of the object.
        """
        return "TDTFile: " + self.filename

    def __repr__(self):
        """
        This function returns a string representation of the object.
        """
        return "TDTFile: " + self.filename

    def get_data(self):
        """
        This function returns the data stored in the class.
        """
        return self.data

    def get_filename(self):
        """
        This function returns the filename stored in the class.
        """
        return self.filename

    def get_epocs(self):
        """
        This function returns the epocs stored in the class.
        """
        return self.data.epocs

    def get_streams(self):
        """
        This function returns the streams stored in the class.
        """
        return self.data.streams

    def get_snips(self):
        """
        This function returns the snips stored in the class.
        """
        return self.data.snips

    def get_scalars(self):
        """
        This function returns the scalars stored in the class.
        """
        return self.data.scalars

    def get_time_ranges(self):
        """
        This function returns the time ranges stored in the class.
        """
        return self.data.time_ranges

    def get_info(self):
        """
        This function returns the info stored in the class.
        """
        return self.data.info

    def save(self, path):
        """
        This function saves the data stored in the class to a TDT file.
        """
        tdt.write_block(self.data, path)
