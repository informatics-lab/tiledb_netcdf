import os

import zarr

from .core import Writer


class ZarrWriter(Writer):
    """Provides a class to write Python objects loaded from NetCDF to zarr."""
    def __init__(self, data_model, array_filepath, array_name=None):
        super().__init__(data_model, array_filepath, array_name, unlimited_dims=None)

        filename = os.path.join(os.path.abspath("."), self.array_filepath, self.array_name)
        self.array_filename = f'{filename}.zarr'
        print(self.array_filename)

        self.group = None
        self.zarray = None

    def create_variable_datasets(self, var_names):
        """
        Create a zarr group - containing data variables and dimensions - for
        a given domain.

        A domain is described by the tuple of dimensions that describe it.

        """
        # Write domain variables and dimensions into group.
        for var_name in var_names:
            data_var = self.data_model.variables[var_name]
            chunks = self.data_model.get_chunks(var_name)
            data_array = self.group.create_dataset(var_name,
                                                shape=data_var.shape,
                                                chunks=chunks,
                                                dtype=data_var.dtype)
            data_array[:] = data_var[...]

            # Set array attributes from ncattrs.
            for ncattr in data_var.ncattrs():
                data_array.attrs[ncattr] = data_var.getncattr(ncattr)

            # Set attribute to specify var's dimensions.
            data_array.attrs['_ARRAY_DIMENSIONS'] = data_var.dimensions

    def create_zarr(self):
        """
        Create a zarr for the contents of `self.data_model`. The grouped
        structure of this zarr is:

            root (filename)
             | - phenom_0
             | - phenom_1
             | - ...
             | - dimension_0
             | - dimension_1
             | - ...
             | - phenom_n
             | - ...

        TODO: add global NetCDF attributes to outermost zarr structure?

        """
        store = zarr.DirectoryStore(self.array_filename)
        self.group = zarr.group(store=store)

        # Write zarr datasets for data variables.
        for domain in self.data_model.domains:
            domain_vars = self.data_model.domain_varname_mapping[domain]
            self.create_variable_datasets(domain_vars)

        # Write zarr datasets for dimension variables.
        keys = list(self.data_model.domain_varname_mapping.keys())
        unique_flat_keys = set([k for domain in keys for k in domain])
        self.create_variable_datasets(unique_flat_keys)

    def append(self, other_data_model, var_name, append_dim):
        """
        Append the contents of other onto self.group, optionally specifying
        a single zarr array to append to with `group_name`.

        If this is not specified, extend all of the arrays in self.group_name
        with all of the arrays found in other. This assumes that the data
        variables in self and other:
          a) are identical
          b) all append along the same dimension.

        Note: append axis is limited to a single axis.

        """
        # Check if the append can go ahead.
        self._append_checker(other_data_model, var_name, append_dim)

        # Work out the index of the append dimension.
        append_axis, append_dim = self._append_dimension(var_name, append_dim)

        with other_data_model.open_netcdf():
            # Append data to the phenomenon.
            other_data_var = other_data_model.variables[var_name]
            getattr(self.group, var_name).append(other_data_var[...], axis=append_axis)

            # Append coordinate values to the append dimension.
            other_dim = other_data_model.variables[append_dim]
            getattr(self.group, append_dim).append(other_dim[...], axis=0)