from collections import namedtuple
from contextlib import contextmanager
from hashlib import md5
import os

import netCDF4
import numpy as np
import tiledb


class NCDataModel(object):

    def __init__(self, netcdf_filename):
        self.netcdf_filename = netcdf_filename
        self._ncds = None

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

        self._data_vars_mapping = None
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

    def dataset_open(self):
        """Check if the dataset has been loaded and is still open."""
        result = False
        if self._ncds is not None:
            result = self._ncds.isopen()
        return result

    @property
    def data_vars_mapping(self):
        if self._data_vars_mapping is None:
            self.data_vars_mapping = {metadata_hash(self, n): [self, n] for n in self.data_var_names}
        return self._data_vars_mapping

    @data_vars_mapping.setter
    def data_vars_mapping(self, value):
        self._data_vars_mapping = value

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
                # If it's a data variable it might also have cell methods.
                if hasattr(variable, 'cell_methods'):
                    self.cell_methods.append(variable.cell_methods)

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


def metadata_hash(data_model, name):
    """
    Produce a completely predictable hash from the metadata of a data model.
    This will make it possible to correctly index an existing array to append
    further data onto the correct index of the existing array.

    The predictable metadata hash of the array is of the form:
        ```name(s)_hash```

    where:
        * `name` is the name(s) of the dataset (CF standard_name if available)
        * `hash` is an md5 hash of a standardised subset of the data model's metadata.

    The metadata that makes up the hash is as follows:
        * name of data variable
        * shape of dataset
        * dimension coordinate names
        * grid_mapping name
        * string of cell methods applied to dataset.

    """
    dims = ",".join(data_model.dim_coord_names)
    grid_mapping = ",".join(data_model.grid_mapping)
    cell_methods = ",".join(data_model.cell_methods)

    to_hash = f"{name}_{dims}_{data_model.shape}_{grid_mapping}_{cell_methods}"
    metadata_hash = md5(to_hash.encode("utf-8")).hexdigest()

    return f"{name}_{metadata_hash}"


class _VarDimLookup(object):
    def __init__(self,
                 primary_data_model,
                 data_vars_mapping=None,
                 variables=True,
                 can_be_none=False):
        self.primary_data_model = primary_data_model
        self.data_vars_mapping = data_vars_mapping
        self.variables = variables
        self.data_model_attr = "variables" if self.variables else "dimensions"
        self.can_be_none = can_be_none

        self._data_var_names = None

    @property
    def data_var_names(self):
        if self._data_var_names is None:
            if self.data_vars_mapping is not None:
                self.data_var_names = list(self.data_vars_mapping.keys())
            else:
                self.data_var_names = []
        return self._data_var_names

    @data_var_names.setter
    def data_var_names(self, value):
        self._data_var_names = value

    def __getitem__(self, keys):
        if keys in self.data_var_names:
            target = self.data_vars_mapping[keys]
        else:
            result = self.primary_data_model
            target = [result, keys]

        target_data_model, original_key = target
        try:
            result = getattr(target_data_model, self.data_model_attr)[original_key]
        except KeyError:
            if self.can_be_none:
                result = None
            else:
                raise
        return result


class NCDataModelGroup(object):
    """
    Combine multiple data model instances (each with only a single data variable)
    into an amalgam data model containing multiple data variables.

    It is assumed (but not currently programatically confirmed) that all data
    model instances are otherwise identical (that is, all other metadata matches),
    and only the data variable changes between instances. Failure to heed this
    limitation will likely lead to broken TileDB / Zarr arrays.

    """
    def __init__(self, data_models):
        self._data_models = data_models

        self._load()

        self._primary_data_model = None
        self._data_var_names = None
        self._data_vars_mapping = None

        self.netcdf_filename = self.primary_data_model.netcdf_filename
        self.variables = _VarDimLookup(self.primary_data_model,
                                       data_vars_mapping=self.data_vars_mapping,
                                       can_be_none=None in self.data_models)
        self.dimensions = _VarDimLookup(self.primary_data_model,
                                        variables=False)

    def __getattr__(self, name):
        """
        Pass on all unhandled attribute get requests to the primary data model,
        which is assumed to be representative of all encapsulated data models
        with regard to other metadata.

        """
        return getattr(self.primary_data_model, name)

    @property
    def data_models(self):
        return self._data_models

    @data_models.setter
    def data_models(self, value):
        self._data_models = value

    @property
    def primary_data_model(self):
        """The 'primary' data model is the first non-None data model in the list."""
        if self._primary_data_model is None:
            result = None
            for data_model in self.data_models:
                if data_model is not None:
                    result = data_model
                    break
            self.primary_data_model = result
        return self._primary_data_model

    @primary_data_model.setter
    def primary_data_model(self, value):
        self._primary_data_model = value

    @property
    def data_var_names(self):
        if self._data_var_names is None:
            self.data_var_names = list(self.data_vars_mapping.keys())
        return self._data_var_names

    @data_var_names.setter
    def data_var_names(self, value):
        self._data_var_names = value

    @property
    def data_vars_mapping(self):
        if self._data_vars_mapping is None:
            self.data_vars_mapping = self._map_data_vars()
        return self._data_vars_mapping

    @data_vars_mapping.setter
    def data_vars_mapping(self, value):
        self._data_vars_mapping = value

    @property
    def scalar_coord_names(self):
        dm_scalar_coords = []
        for dm in self.data_models:
            dm_scalar_coords += dm.scalar_coord_names
        return list(set(dm_scalar_coords))

    def _map_data_vars(self):
        """Create a mapping of data variable names to the data model supplying that data variable."""
        dv_mapping = {}
        for dm in self.data_models:
            if dm is not None:
                dv_mapping.update(dm.data_vars_mapping)
        return dv_mapping

    @contextmanager
    def open_netcdf(self):
        try:
            self.open()
            yield
        finally:
            self.close()

    def _load(self):
        """Ensure all data models passed to the constructor are DataModel objects or None."""
        dms = []
        for data_model in self.data_models:
            if isinstance(data_model, str):
                try:
                    dm = NCDataModel(data_model)
                    dm.populate()
                except (FileNotFoundError, AttributeError):
                    dm = None
                dms.append(dm)
            else:
                dms.append(data_model)
        self.data_models = dms

    def open(self):
        for data_model in self.data_models:
            if data_model is not None:
                data_model.open()

    def close(self):
        for data_model in self.data_models:
            if data_model is not None:
                data_model.close()

    def dataset_open(self):
        """
        This 'dataset' (specifically a group of datasets) can only be considered
        'open' if all the datasets that comprise it are open.

        """
        return all([dm.dataset_open() for dm in self.data_models if dm is not None])