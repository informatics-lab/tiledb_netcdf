import os

import iris
import tiledb
import xarray as xr
import zarr


class Reader(object):
    """
    Abstract reader class that defines the API.

    """
    def __init__(self, input_filepath, output_filepath):
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath

        self._artifact = None

    @property
    def artifact(self):
        return self._artifact

    @artifact.setter
    def artifact(self, value):
        self._artifact = value

    def to_iris(self):
        """Convert the input to an Iris cube or cubelist, depending on input."""
        raise NotImplementedError

    def to_xarray(self):
        """Convert the input to an Xarray dataset."""
        raise NotImplementedError


class TDBReader(Reader):
    pass


class ZarrReader(Reader):
    def __init__(self, input_filepath, output_filepath):
        super().__init__(input_filepath, output_filepath)

    def to_iris(self):
        intermediate = self.to_xarray()
        self.artifact = intermediate.to_iris()
        return self.artifact

    def to_xarray(self):
        self.artifact = xr.open_zarr(self.input_filepath)
        return self.artifact