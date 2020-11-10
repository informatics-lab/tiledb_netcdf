class Reader(object):
    """
    Abstract reader class that defines the API.

    TODO replace all os usages with tiledb ls'.

    """
    horizontal_coord_names = ['latitude',
                              'longitude',
                              'grid_latitude',
                              'grid_longitude',
                              'projection_x_coordinate',
                              'projection_y_coordinate']
    def __init__(self, array_filepath):
        self.array_filepath = array_filepath

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