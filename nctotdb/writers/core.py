import os

import numpy as np
import tiledb

from .. import utils


class Writer(object):
    """
    An abstract base for specific writers to write the contents
    of a NetCDF file to a different format.

    """
    def __init__(self, data_model,
                 array_filepath=None, container=None, array_name=None,
                 unlimited_dims=None, ctx=None):
        self._data_model = data_model
        self.array_filepath = array_filepath
        self.container = container  # Azure container name.
        self.unlimited_dims = unlimited_dims
        self.ctx = ctx  # TileDB Context object.

        self._scalar_unlimited = None

        # Need either a local filepath or a remote container.
        utils.ensure_filepath_or_container(self.array_filepath, self.container)

        self._array_name = array_name
        if self._array_name is None:
            self.array_name = os.path.basename(os.path.splitext(self.data_model.netcdf_filename)[0])
        else:
            self.array_name = self._array_name
        self.array_path = utils.filepath_generator(self.array_filepath,
                                                   self.container,
                                                   self.array_name,
                                                   ctx=self.ctx)

    @property
    def data_model(self):
        if not self._data_model.dataset_open():
            self._data_model.open()
        return self._data_model

    def _all_coords(self, variable):
        dim_coords = list(variable.dimensions)
        other_coords = variable.coordinates.split(' ')
        return dim_coords + other_coords

    def _get_dim_coord_names(self, var_name):
        """
        Figure out names of the dimension-describing coordinates for this array,
        including the promoted append-dimension scalar coordinate if necessary.

        """
        dim_coord_names = self.data_model.variables[var_name].dimensions
        if self._scalar_unlimited is not None:
            # `dim_coord_names` is a tuple...
            dim_coord_names = (self._scalar_unlimited,) + dim_coord_names
        return dim_coord_names

    def _append_checker(self, other_data_model, var_name, append_dim):
        """Checks to see if an append operation can go ahead."""
        # Sanity checks: is the var name in both self, other, and the tiledb?
        assert var_name in self.data_model.data_var_names, \
            f'Variable name {var_name!r} not found in this data model.'
        assert var_name in other_data_model.data_var_names, \
            f'Variable name {var_name!r} not found in other data model.'

        self_var = self.data_model.variables[var_name]
        self_var_coords = self._all_coords(self_var)
        # Is the append dimension valid?
        assert append_dim in self_var_coords, \
            f'Dimension {append_dim!r} not found in this data model.'

        with other_data_model.open_netcdf():
            other_var = other_data_model.variables[var_name]
            other_var_coords = self._all_coords(other_var)
            # Is the append dimension valid?
            assert append_dim in other_var_coords, \
                f'Dimension {append_dim!r} not found in other data model.'

    def _append_dimension(self, var_name, append_desc):
        """Determine the name and index of the dimension for the append operation."""
        if not isinstance(append_desc, int):
            # Find the append axis from the dimension name.
            append_axis = self.data_model.variables[var_name].dimensions.index(append_desc)
            append_dim = append_desc
        else:
            # Find the append dimension name from the axis.
            append_axis = append_desc
            append_dim = self.data_model.dimensions[append_axis]
        return append_axis, append_dim

    def _fill_missing_points(self, coord_array_path, coord_array_name, verbose=False):
        """
        If one or more indices along the append axis are missing spatial points, we
        end up with `NaN`s in the resultant coordinate array. This prevents loading
        into Iris (as we cannot make a monotonic coordinate array).

        Fill such missing points with interpolated point values so that Iris can load
        the dataset, if with missing data points still. Use a simple custom 1D
        interpolator as the SciPy and NumPy offerings cannot handle NaN values.

        """
        with tiledb.open(coord_array_path, 'r', ctx=self.ctx) as D:
            ned = D.nonempty_domain()[0]
            coord_points = D[ned[0]:ned[1]][coord_array_name]

        missing_points, = np.nonzero(np.isnan(coord_points))
        if len(missing_points):
            if verbose:
                print(f'{len(missing_points)} points to fill in {coord_array_name!r}.')

            ind_points = np.arange(len(coord_points))
            coord_steps = np.unique(np.diff(coord_points))
            # Expects only a single non-NaN step (i.e. monotonicity).
            numeric_step, = coord_steps[np.nonzero(~np.isnan(coord_steps))]

            # Interpolate to fill the missing points.
            vec_interp = np.vectorize(utils.fillnan)
            coord_points[missing_points] = vec_interp(ind_points[missing_points],
                                                      coord_points[0],
                                                      numeric_step)

            # Write the whole filled array back to the TileDB coord array.
            with tiledb.open(coord_array_path, 'w', ctx=self.ctx) as D:
                D[ned[0]:ned[1]] = coord_points
        else:
            if verbose:
                print(f'No missing points in {coord_array_name!r}, nothing to do.')
