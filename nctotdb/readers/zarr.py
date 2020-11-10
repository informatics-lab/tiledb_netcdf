import xarray as xr

from .core import Reader


class ZarrReader(Reader):
    def __init__(self, array_filepath):
        super().__init__(array_filepath)

    def to_iris(self):
        intermediate = self.to_xarray()
        self.artifact = intermediate.to_iris()
        return self.artifact

    def to_xarray(self):
        self.artifact = xr.open_zarr(self.array_filepath)
        return self.artifact
