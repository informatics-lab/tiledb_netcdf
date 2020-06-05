from collections import namedtuple
from contextlib import contextmanager
import os

import netCDF4
import numpy as np
import tiledb


class NCDataModel(object):

    def __init__(self, netcdf_filename):
        self.netcdf_filename = netcdf_filename

        self.data_var_names = []
        self.dim_coord_names = []
        self.scalar_coord_names = []
        self.aux_coord_names = []
        self.bounds = []
        self.grid_mapping = []
        self.cell_methods = []
        self.cell_measures = []
        self.unlimited_dim_coords = []

        self.domains = []
        self.domain_varname_mapping = None
        self.varname_domain_mapping = None
        self.shape = None
        self.chunks = None

        self._nc_loaded = False
        self._classified = False

    def open(self):
        # Open the NC file and retrieve key elements of it.
        self._ncds = netCDF4.Dataset(self.netcdf_filename, mode='r')
        self.dimensions = self._ncds.dimensions
        self.variables = self._ncds.variables
        # Also set derived attributes, if not already done.
        if not self._nc_loaded:
            self.dimension_names = list(self.dimensions.keys())
            self.variable_names = list(self.variables.keys())
            self.ncattrs = {key: self._ncds.getncattr(key) for key in self._ncds.ncattrs()}
            self._nc_loaded = True

    def close(self):
        self._ncds.close()

    @contextmanager
    def open_netcdf(self):
        try:
            self.open()
            yield
        finally:
            self.close()

    def populate(self):
        with self.open_netcdf():
            self.classify_variables()
            self.get_metadata()

    def classify_variables(self):
        """
        Classify all of the NetCDF variables as one of the following:
          * Data Variable
          * Dimension Coordinates
          * Auxiliary Coordinates
          * Scalar Coordinates
          * Grid Mapping Variable
          * Bounds
          * Cell Measures
          * (Cell Methods)
          * Unlimited Dim Coords
          * Something else.

        """
        # Classify unlimited dimensions.
        self.unlimited_dim_coords = [name for name in self.dimension_names
                                     if self.dimensions[name].isunlimited()]

        classified_vars = []
        for variable_name, variable in self.variables.items():
            # Check if this variable is a grid mapping variable.
            if hasattr(variable, 'grid_mapping_name'):
                self.grid_mapping.append(variable_name)
                classified_vars.append(variable_name)

            # Check if this variable is a data variable.
            elif hasattr(variable, 'coordinates'):
                self.data_var_names.append(variable_name)
                classified_vars.append(variable_name)

            # Check if this variable is a coordinate - dimension or aux.
            elif hasattr(variable, 'dimensions'):
                if variable_name in self.dimension_names:
                    # This is a dimension coordinate.
                    self.dim_coord_names.append(variable_name)
                elif not len(variable.dimensions):
                    # This is a scalar coordinate.
                    self.scalar_coord_names.append(variable_name)
                elif variable_name.endswith('bnds'):
                    # Check if it's a coordinate bounds.
                    self.bounds.append(variable_name)
                else:
                    # This is an auxiliary coordinate.
                    self.aux_coord_names.append(variable_name)
                classified_vars.append(variable_name)

            # Check if it's a cell measure variable.
            elif hasattr(variable, 'cell_measures'):
                self.cell_measures.append(variable_name)
                classified_vars.append(variable_name)
                classified_vars.append(variable_name)

            # TODO: check if it's a cell method variable
#             elif hasattr(variable, 'cell_measures'):
#                 pass

        # What have we still missed?
        unclassified_vars = list(set(self.variable_names) - set(classified_vars))
        if len(unclassified_vars):
            # We're not trying again, so just print them.
            print(f'Unclassified vars: {unclassified_vars}')

        # We've now classified this NC file.
        self._classified = True

    def get_chunks(self, data_var_name, max_contiguous_dims=3):
        """
        Get chunks for a named data variable `data_var_name`.

        Chunking can be tricky as 'contiguous' is a valid NetCDF
        chunking strategy (i.e. there's only one chunk and the data is
        contiguous on disk). In this case we want the chunking to match
        the shape, which is an equivalent statement.
        One heuristic we apply is that for ndim > 3 the chunking of all
        leading dimensions is [1,] to avoid very large chunks.

        """
        data_var = self.variables[data_var_name]
        chunks = data_var.chunking()
        if chunks == 'contiguous':
            shape = data_var.shape
            data_ndim = len(shape)
            overflow_dims = data_ndim - max_contiguous_dims
            if data_ndim > max_contiguous_dims:
                # More than 3D so chunk along outer (leading) dimension
                # to keep chunk sizes down.
                leading_chunksizes = [1] * overflow_dims
                chunks = tuple(leading_chunksizes + list(shape[overflow_dims:]))
            else:
                chunks = shape
        return chunks

    def get_domains(self):
        """
        Determine the unique set of domains described in all the variables in the dataset.
        Here domain means the set of dimensions that describe a variable, such as
        (time, lon, lat), which itself is a super-domain of the domain (lon, lat). As such
        any variable on the domain (lon, lat) can be fitted into the super-domain.

        If there is only one variable, this can be skipped.

        """
        # Work out the shapes of each variable. The most enclosing domains will have the highest ndim.
        ndims = np.array([len(self.variables[var_name].shape) for var_name in self.data_var_names])
        max_ndim = max(ndims)
        # Get the variables that describe the most enclosing domains (super domains).
        super_domain_vars = np.array(self.data_var_names)[ndims == max_ndim]
        domain_dims = [self.variables[var_name].dimensions for var_name in super_domain_vars]
        # Get the unique super domains.
        super_domains = list(set(domain_dims))
        # Get the variables that haven't been checked for domain inclusion.
        undomained_vars = set(self.data_var_names) - set(super_domains)

        # Check for super domains with fewer than the maximum ndim.
        for var_name in self.data_var_names:
            dims = self.variables[var_name].dimensions
            partial_coverage = [set(dims) - set(domain) for domain in super_domains]
            if all(partial_coverage):
                # This particular domain isn't fully represented by any existing super-domain,
                # so is its own new super-domain.
                super_domains.append(dims)

        self.domains = super_domains

    def domain_for_var(self):
        """
        Produce a mapping between each variable name and the super-domain that describes it.

        TODO: the first super-domain will be chosen. For a 2D domain this could result in
        incorrect classification if the first matching super-domain doesn't match the scalar
        coords of the variable.

        """
        # Build a dictionary mapping var_name to covered dimensions for undomained variables.
        name_dims_mapping = {var_name: self.variables[var_name].dimensions
                             for var_name in self.data_var_names}

        # Filling this is our target.
        name_domain_mapping = {k: [] for k in self.domains}
        # TODO: use an itertools product instead.
        for var_name, dims in name_dims_mapping.items():
            valid_domain = None
            for domain in self.domains:
                # Check if all the dims are included in the domain. If they are, then...
                if len(set(dims) - set(set(dims) & set(domain))) == 0:
                    # ... this is a valid super domain.
                    valid_domain = domain
                    break
            name_domain_mapping[valid_domain].append(var_name)
        self.domain_varname_mapping = name_domain_mapping

    def get_metadata(self):
        """
        Set extra metadata now that we've classified the dataset variables, notably:

          * the domain(s)
          * chunks
          * shape

        The last two are only done if there is just a single variable, otherwise they are
        left as `None`, and downstream writer class is responsible for setting these on a
        per-domain or per-data_variable basis.

        """
        n_data_vars = len(self.data_var_names)
        if n_data_vars == 1:
            # Only one data var so we can set domain, chunks and shape from it.
            data_var = self.variables[self.data_var_names[0]]
            self.domains.append(data_var.dimensions)
            self.domain_varname_mapping = {data_var.dimensions: [self.data_var_names[0]]}
            self.chunks = self.get_chunks(self.data_var_names[0])
            self.shape = data_var.shape
        elif n_data_vars > 1:
            # Multiple data vars means we need to set domains and variable-domain mapping.
            self.get_domains()
            self.domain_for_var()
        else:
            # No data var means trouble.
            raise ValueError(f'Expected to find at least one data var, but found {n_data_vars}.')

        # Also create the inverse mapping of var_name --> domain.
        self.varname_domain_mapping = {vi: k for k, v in self.domain_varname_mapping.items() for vi in v}
