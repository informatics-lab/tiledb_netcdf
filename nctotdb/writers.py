from collections import defaultdict
import copy
import logging
import os

import numpy as np
import tiledb
import zarr

from .data_model import NCDataModel


class Writer(object):
    """
    An abstract base for specific writers to write the contents
    of a NetCDF file to a different format.

    """
    def __init__(self, data_model, array_filepath,
                 array_name=None, unlimited_dims=None):
        self.data_model = data_model
        self.array_filepath = array_filepath
        self.unlimited_dims = unlimited_dims

        self._array_name = array_name
        if self._array_name is None:
            self.array_name = os.path.basename(os.path.splitext(self.data_model.netcdf_filename)[0])
        else:
            self.array_name = self._array_name

    def _all_coords(self, variable):
        dim_coords = list(variable.dimensions)
        other_coords = variable.coordinates.split(' ')
        return dim_coords + other_coords

    def _append_checker(self, other_data_model, var_name, append_dim):
        """Checks to see if an append operation can go ahead."""
        # Sanity checks: is the var name in both self, other, and the tiledb?
        assert var_name in self.data_model.data_var_names, f'Variable name {var_name!r} not found in this data model.'
        assert var_name in other_data_model.data_var_names, f'Variable name {var_name!r} not found in other data model.'

        self_var = self.data_model.variables[var_name]
        self_var_coords = self._all_coords(self_var)
        other_var = other_data_model.variables[var_name]
        other_var_coords = self._all_coords(other_var)
        # And is the append dimension valid?
        assert append_dim in self_var_coords, f'Dimension {append_dim!r} not found in this data model.'
        assert append_dim in other_var_coords, f'Dimension {append_dim!r} not found in other data model.'

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


class TDBWriter(Writer):
    """
    Provides a class to write Python objects loaded from NetCDF to TileDB.

    Data Model: an instance of `NCDataModel` supplying data from a NetCDF file.
    Filepath: the filepath to save the tiledb array at.

    """
    def __init__(self, data_model, array_filepath,
                 array_name=None, unlimited_dims=None):
        super().__init__(data_model, array_filepath, array_name, unlimited_dims)
        if self.unlimited_dims is None:
            self.unlimited_dims = []

    def _public_domain_name(self, domain):
        domain_index = self.data_model.domains.index(domain)
        return f'domain_{domain_index}'

    def _create_tdb_directory(self, group_dirname):
        """
        Create an on-filesystem directory for a tiledb group if it does
        not exist, and ignore the error if it does.

        """
        try:
            os.makedirs(group_dirname)
        except FileExistsError:
            pass

    def _create_tdb_dim(self, dim_name, coords):
        dim_coord = self.data_model.variables[dim_name]
        chunks = self.data_model.get_chunks(dim_name)

        # TODO: work out nD coords (although a DimCoord will never be nD).
        dim_coord_len, = dim_coord.shape

        # Set the tdb dimension dtype to `int64` regardless of input.
        # Dimensions must have int indices for dense array schemas.
        # All tdb dims in a domain must have exactly the same dtype.
        dim_dtype = np.int64

        # Sort out the domain, based on whether the dim is unlimited,
        # or whether it was specified that it should be by `self.unlimited_dims`.
        if dim_name in self.unlimited_dims:
            domain_max = np.iinfo(dim_dtype).max - dim_coord_len
        elif dim_name in self.data_model.unlimited_dim_coords:
            domain_max = np.iinfo(dim_dtype).max - dim_coord_len
        else:
            domain_max = dim_coord_len

        # Modify the name of the dimension if this dimension describes the domain
        # for a dim coord array.
        # Array attrs and dimensions must have different names.
        if coords:
            dim_name = f'{dim_name}_coord'

        return tiledb.Dim(name=dim_name,
                          domain=(0, domain_max),
                          tile=chunks,
                          dtype=dim_dtype)

    def create_domain_arrays(self, domain_vars, group_dirname, coords=False):
        """Create one single-attribute array per data var in this NC domain."""
        for var_name in domain_vars:
            # Set dims for the enclosing domain.

            data_var = self.data_model.variables[var_name]
            data_var_dims = data_var.dimensions
            array_dims = [self._create_tdb_dim(dim_name, coords) for dim_name in data_var_dims]
            tdb_domain = tiledb.Domain(*array_dims)

            # Get tdb attributes.
            attr = tiledb.Attr(name=var_name, dtype=data_var.dtype)

            # Create the URI for the array.
            array_filename = os.path.join(group_dirname, var_name)
            # Create an empty array.
            schema = tiledb.ArraySchema(domain=tdb_domain, sparse=False, attrs=[attr])
            tiledb.Array.create(array_filename, schema)

    def _array_indices(self, shape, start_index):
        """Set the array indices to write the array data into."""
        return _array_indices(shape, start_index)

    def populate_array(self, var_name, data_var, group_dirname,
                       start_index=None, write_meta=True):
        """Write the contents of a netcdf data variable into a tiledb array."""
        # Get the data variable and the filename of the array to write to.
        var_name = data_var.name
        array_filename = os.path.join(group_dirname, var_name)

        # Write to the array.
        write_array(array_filename, data_var, start_index=start_index)
        if write_meta:
            with tiledb.open(array_filename, 'w') as A:
                # Set tiledb metadata from data var ncattrs.
                for ncattr in data_var.ncattrs():
                    A.meta[ncattr] = data_var.getncattr(ncattr)
                # Add metadata describing whether this is a coord or data var.
                if var_name in self.data_model.data_var_names:
                    # A data array gets a `dataset` key in the metadata dictionary,
                    # which defines all the data variables in it.
                    A.meta['dataset'] = var_name
                    # Define this as not being a multi-attr array.
                    A.meta['multiattr'] = False
                    # XXX: can't add list or tuple as values to metadata dictionary...
                    dim_coord_names = self.data_model.variables[var_name].dimensions
                    A.meta['dimensions'] = ','.join(n for n in dim_coord_names)
                elif var_name in self.data_model.dim_coord_names:
                    # A dim coord gets a `coord` key in the metadata dictionary,
                    # value being the name of the coordinate.
                    A.meta['coord'] = self.data_model.dimensions[var_name].name
                else:
                    # Don't know how to handle this. It might be an aux or scalar
                    # coord, but we're not currently writing TDB arrays for them.
                    pass

    def populate_domain_arrays(self, domain_vars, group_dirname):
        """Populate all arrays with data from netcdf data vars within a tiledb group."""
        for var_name in domain_vars:
            data_var = self.data_model.variables[var_name]
            self.populate_array(var_name, data_var, group_dirname)

    def create_domains(self):
        """
        We need to create one TDB group per data variable in the data model,
        organised by domain.

        """
        for domain in self.data_model.domains:
            # Get the data and coord variables in this domain.
            domain_vars = self.data_model.domain_varname_mapping[domain]
            # Defined for the sake of clarity (each `domain` is a list of its dim coords).
            domain_coords = domain

            # Create group.
            domain_name = self._public_domain_name(domain)
            group_dirname = os.path.join(self.array_filepath, self.array_name, domain_name)
            # TODO why is this necessary? Shouldn't tiledb create if this dir does not exist?
            self._create_tdb_directory(group_dirname)
            # TODO it would be good to write the domain's dim names into the group meta.
            tiledb.group_create(group_dirname)

            # Create and write arrays for each domain-describing coordinate.
            self.create_domain_arrays(domain_coords, group_dirname, coords=True)
            self.populate_domain_arrays(domain_coords, group_dirname)

            # Get data vars in this domain and create an array for the domain.
            self.create_domain_arrays(domain_vars, group_dirname)
            # Populate this domain's array.
            self.populate_domain_arrays(domain_vars, group_dirname)

    def append(self, others, var_name, append_dim,
              logfile=None, parallel=False, verbose=False):
        """
        Append extra data as described by the contents of `others` onto
        an existing TileDB array along the axis defined by `append_dim`.

        Notes:
          * extends one dimension only
          * cannot create new dimensions, only extend existing dimensions

        TODO support multiple axis appends.
        TODO check if there's already data at the write inds and add an overwrite?

        """
        if logfile is not None:
            logging.basicConfig(filename=logfile,
                                level=logging.DEBUG,
                                format='%(asctime)s %(message)s',
                                datefmt='%d/%m/%Y %H:%M:%S')

        make_data_model = False
        # Check what sort of thing `others` is.
        if isinstance(others, NCDataModel):
            others = [others]
        elif isinstance(others, str):
            others = [others]
            make_data_model = True
        else:
            other = others[0]
            if isinstance(other, str):
                make_data_model = True

        append_axis, append_dim = self._append_dimension(var_name, append_dim)

        self_dim_var = self.data_model.variables[append_dim]
        self_dim_points = copy.copy(self_dim_var[:])
        self_dim_start, self_dim_stop, self_step = _dim_points(self_dim_points)
        self_ind_start, self_ind_stop = _dim_inds(self_dim_points,
                                                  [self_dim_start, self_dim_stop])

        # Get domain for var_name and tiledb array path.
        domain = self.data_model.varname_domain_mapping[var_name]
        domain_name = f'domain_{self.data_model.domains.index(domain)}'
        domain_path = os.path.join(self.array_filepath, self.array_name, domain_name)

        # For multidim / multi-attr appends this will be more complex.
        common_job_args = [domain_path, var_name, append_axis, append_dim,
                           self_ind_stop, self_dim_stop, self_step,
                           make_data_model, verbose]
        job_args = [[other] + common_job_args for other in others]

        if parallel:
            import dask.bag as db
            bag_of_jobs = db.from_sequence(job_args)
            bag_of_jobs.map(_make_tile_helper).compute()
        else:
            for i, args in enumerate(job_args):
                args += [i, len(job_args)]
                _make_tile_helper(args)

        if len(others) > 10:
            # Consolidate at the end of the append operations to make the resultant
            # array more performant.
            config = tiledb.Config({"sm.consolidation.steps": 1000})
            ctx = tiledb.Ctx(config)
            tiledb.consolidate(os.path.join(domain_path, var_name), ctx=ctx)


class MultiAttrTDBWriter(TDBWriter):
    """
    Provides a class to write Python objects loaded from NetCDF to TileDB.

    Data Model: an instance of `NCDataModel` supplying data from a NetCDF file.
    Filepath: the filepath to save the tiledb array at.

    """
    def __init__(self, data_model, array_filepath,
                 array_name=None, unlimited_dims=None, domain_separator=','):
        super().__init__(data_model, array_filepath, array_name, unlimited_dims)

        self.domain_separator = domain_separator
        self._domains_mapping = None

    @property
    def domains_mapping(self):
        if self._domains_mapping is None:
            self._make_shape_domains()
        return self._domains_mapping

    @domains_mapping.setter
    def domains_mapping(self, value):
        self._domains_mapping = value

    def _public_domain_name(self, dimensions, separator):
        """
        A common method for determining the domain name for a given data variable,
        based on its dimensions.

        """
        return separator.join(dimensions)

    def _get_attributes(self, data_var):
        metadata = {}
        for ncattr in data_var.ncattrs():
            metadata[ncattr] = data_var.getncattr(ncattr)
        return metadata

    def _multi_attr_metadata(self, data_var_names):
        if len(data_var_names) == 1:
            data_var = self.data_model.variables[data_var_names[0]]
            metadata = self._get_attributes(data_var)
        else:
            metadata = {}
            for var_name in data_var_names:
                data_var = self.data_model.variables[var_name]
                data_var_meta = self._get_attributes(data_var)
                # XXX TileDB does not support dict in array metadata...
                for key, value in data_var_meta.items():
                    metadata[f'{key}__{var_name}'] = value
        return metadata

    def _make_shape_domains(self):
        """
        Make one domain for each unique combination of shape and dimension variables.

        We need to make this set of domains for the multi-attr case as a limitation
        in TileDB means that all attrs in an array must be written at the same time,
        which also means the indexing for all attrs to be written must be the same.

        """
        shape_domains = []
        for data_var_name in self.data_model.data_var_names:
            dimensions = self.data_model.variables[data_var_name].dimensions
            domain_string = self._public_domain_name(dimensions, self.domain_separator)
            shape_domains.append((domain_string, data_var_name))

        domains_mapping = defaultdict(list)
        for domain, data_var_name in shape_domains:
            domains_mapping[domain].append(data_var_name)
        self.domains_mapping = domains_mapping

    def populate_multiattr_array(self, data_array_name, data_var_names, group_dirname,
                                 start_index=None, write_meta=True):
        """Write the contents of multiple data variables into a multi-attr TileDB array."""
        array_filename = os.path.join(group_dirname, data_array_name)

        # Write to the array.
        data_vars = [self.data_model.variables[name] for name in data_var_names]
        write_multiattr_array(array_filename, data_vars, start_index=start_index)
        if write_meta:
            multi_attr_metadata = self._multi_attr_metadata(data_var_names)
            with tiledb.open(array_filename, 'w') as A:
                # Set tiledb metadata from data var ncattrs.
                for key, value in multi_attr_metadata.items():
                    A.meta[key] = value
                # A data array gets a `dataset` key in the metadata dictionary,
                # which defines all the data variables in it.
                A.meta['dataset'] = ','.join(data_var_names)
                # Define this as being a multi-attr array.
                A.meta['multiattr'] = True
                # XXX: can't add list or tuple as values to metadata dictionary...
                dim_coord_names = self.data_model.variables[data_var_names[0]].dimensions
                A.meta['dimensions'] = ','.join(n for n in dim_coord_names)

    def create_multiattr_array(self, domain_var_names, domain_dims,
                               group_dirname, data_array_name):
        """Create one multi-attr TileDB array with an attr for each data variable."""
        # Create dimensions and domain for the multi-attr array.
        array_dims = [self._create_tdb_dim(dim_name, coords=False) for dim_name in domain_dims]
        tdb_domain = tiledb.Domain(*array_dims)

        # Set up the multiple attrs for the array.
        attrs = []
        for var_name in domain_var_names:
            dtype = self.data_model.variables[var_name].dtype
            attr = tiledb.Attr(name=var_name, dtype=dtype)
            attrs.append(attr)

        # Create the URI for the array.
        array_filename = os.path.join(group_dirname, data_array_name)
        # Create an empty array.
        schema = tiledb.ArraySchema(domain=tdb_domain, sparse=False, attrs=attrs)
        tiledb.Array.create(array_filename, schema)

    def create_domains(self, data_array_name='data'):
        """
        Create one TileDB domain for each unique shape / dimensions combination
        in the input Data Model. Each domain will contain:
          * one multi-attr array, where the attrs are all the data variables described
            by this combination of dimensions, and
          * one array for each of the dimension-describing coordinates for this
            combination of dimensions.

        """
        self._make_shape_domains()
        
        for domain_name, domain_var_names in self.domains_mapping.items():
            domain_coord_names = domain_name.split(self.domain_separator)

            # Create group.
            group_dirname = os.path.join(self.array_filepath, self.array_name, domain_name)
            # TODO why is this necessary? Shouldn't tiledb create if this dir does not exist?
            self._create_tdb_directory(group_dirname)
            # TODO it would be good to write the domain's dim names into the group meta.
            tiledb.group_create(group_dirname)

            # Create and write arrays for each domain-describing coordinate.
            self.create_domain_arrays(domain_coord_names, group_dirname, coords=True)
            self.populate_domain_arrays(domain_coord_names, group_dirname)

            # Get data vars in this domain and create and populate a multi-attr array.
            self.create_multiattr_array(domain_var_names, domain_coord_names,
                                        group_dirname, data_array_name)
            self.populate_multiattr_array(data_array_name, domain_var_names, group_dirname)

    def _make_tile_helper(self, args, kwargs):
        other, domain_names, data_array_name, append_dim, *other_args = args
        verbose = False
        if kwargs is not None:
            verbose = True
            job_no = kwargs['job_no'] + 1
            n_jobs = kwargs['n_jobs']
            n_domains = len(domain_names)

        other_data_model = NCDataModel(other)
        other_data_model.classify_variables()
        other_data_model.get_metadata()

        for n, domain_name in enumerate(domain_names):
            if verbose:
                fn = other_data_model.netcdf_filename
                print(f'Processing {fn}...  ({job_no}/{n_jobs}, domain {n+1}/{n_domains})', end="\r")

            append_axis = domain_name.split(self.domain_separator).index(append_dim)
            domain_path = os.path.join(self.array_filepath, self.array_name, domain_name)
            array_var_names = self.domains_mapping[domain_name]
            _make_multiattr_tile(other_data_model, domain_path, data_array_name,
                                 array_var_names, append_axis, append_dim, *other_args)

    def append(self, others, append_dim, data_array_name,
              logfile=None, parallel=False, verbose=False):
        """
        Append extra data as described by the contents of `others` onto
        an existing TileDB array along the axis defined by `append_dim`.

        Notes:
          * extends one dimension only
          * cannot create new dimensions, only extend existing dimensions

        TODO support multiple axis appends.
        TODO check if there's already data at the write inds and add an overwrite?

        """
        if logfile is not None:
            logging.basicConfig(filename=logfile,
                                level=logging.DEBUG,
                                format='%(asctime)s %(message)s',
                                datefmt='%d/%m/%Y %H:%M:%S')

        make_data_model = False
        # Check what sort of thing `others` is.
        if isinstance(others, NCDataModel):
            others = [others]
        elif isinstance(others, str):
            others = [others]

        # Check all domains for including the append dimension.
        domain_names = [d for d in self.domains_mapping.keys()
                        if append_dim in d.split(self.domain_separator)]

        # Get starting dimension points and offsets.
        self_dim_var = self.data_model.variables[append_dim]
        self_dim_points = copy.copy(self_dim_var[:])
        self_dim_start, self_dim_stop, self_step = _dim_points(self_dim_points)
        self_ind_start, self_ind_stop = _dim_inds(self_dim_points,
                                                  [self_dim_start, self_dim_stop])

        # For multidim / multi-attr appends this will be more complex.
        common_job_args = [domain_names, data_array_name, append_dim,
                           self_ind_stop, self_dim_stop, self_step]
        job_args = [[other] + common_job_args for other in others]

        if parallel:
            # import dask.bag as db
            # bag_of_jobs = db.from_sequence(job_args)
            # bag_of_jobs.map(_make_tile_helper).compute()
            raise NotImplementedError
        else:
            for i, args in enumerate(job_args):
                kwargs = None
                if verbose:
                    kwargs = {'job_no': i, 'n_jobs': len(job_args)}
                self._make_tile_helper(args, kwargs)

        if len(others) > 10:
            # Consolidate at the end of the append operations to make the resultant
            # array more performant.
            config = tiledb.Config({"sm.consolidation.steps": 1000})
            ctx = tiledb.Ctx(config)
            tiledb.consolidate(os.path.join(domain_path, var_name), ctx=ctx)


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

        # Append data to the phenomenon.
        other_data_var = other_data_model.variables[var_name]
        getattr(self.group, var_name).append(other_data_var[...], axis=append_axis)

        # Append coordinate values to the append dimension.
        other_dim = other_data_model.variables[append_dim]
        getattr(self.group, append_dim).append(other_dim[...], axis=0)

        
###################################################################################
#                                                                                 #
# Remove these functions from `TDBWriter` because most of them are static and it  #
# might make tiling in parallel possible!                                         #
#                                                                                 #
###################################################################################


def _array_indices(shape, start_index):
    """Set the array indices to write the array data into."""
    if isinstance(start_index, int):
        start_index = [start_index] * len(shape)

    array_indices = []
    for dim_len, start_ind in zip(shape, start_index):
        array_indices.append(slice(start_ind, dim_len+start_ind))
    return tuple(array_indices)


def write_array(array_filename, data_var, start_index=None):
    """Write to the array."""
    if start_index is None:
        start_index = 0
        shape = data_var.shape
        write_indices = _array_indices(shape, start_index)
    else:
        write_indices = start_index

    # Write netcdf data var contents into array.
    with tiledb.open(array_filename, 'w') as A:
        A[write_indices] = data_var[...]


def write_multiattr_array(array_filename, data_vars, start_index=None):
    """Write to each attr in the array."""
    if start_index is None:
        start_index = 0
        shape = data_vars[0].shape  # All data vars *must* have the same shape for writing...
        write_indices = _array_indices(shape, start_index)
    else:
        write_indices = start_index

    # Write netcdf data var contents into array.
    with tiledb.open(array_filename, 'w') as A:
        A[write_indices] = {data_var.name: data_var[...] for data_var in data_vars}


def _dim_inds(dim_points, spatial_inds, offset=0):
    """Convert coordinate values to index space."""
    return [list(dim_points).index(si) + offset for si in spatial_inds]


def _dim_points(points):
    """Convert a dimension variable (coordinate) points to index space."""
    start, stop = points[0], points[-1]
    step, = np.unique(np.diff(points))
    return start, stop, step


def _dim_offsets(dim_points, self_stop_ind, self_stop, self_step, scalar=False):
    """
    Calculate the offset along a dimension given by `var_name` between self
    and other.

    """
    if scalar:
        other_start = dim_points
        spatial_inds = [other_start, other_start]  # Fill the nonexistent `stop` with a blank.
    else:
        other_start, other_stop, other_step = _dim_points(dim_points)
        assert self_step == other_step, "Step between coordinate points is not equal."
        spatial_inds = [other_start, other_stop]

    points_offset = other_start - self_stop
    inds_offset = int(points_offset / self_step) + self_stop_ind

    i_start, _ = _dim_inds(dim_points, spatial_inds, inds_offset)
    return i_start


def _progress_report(other_data_model, verbose, i, total):
    """A helpful printout of append progress."""
    # XXX Not sure about this when called from a bag...
    if verbose and i is not None and num is not None:
        fn = other_data_model.netcdf_filename
        ct = i + 1
        pc = 100 * (ct / total)
        print(f'Processing {fn}...  ({ct}/{total}, {pc:0.1f}%)', end="\r")


def _make_tile_helper(args):
    """Helper method to call from a `map` operation and unpack the args."""
    _make_tile(*args)


def _make_multiattr_tile(other_data_model, domain_path, data_array_name,
                         var_names, append_axis, append_dim,
                         self_ind_stop, self_dim_stop, self_step):
    """Process appending a single tile to `self`, per domain."""
    other_data_vars = [other_data_model.variables[var_name] for var_name in var_names]
    data_var_shape  = other_data_vars[0].shape
    other_dim_var = other_data_model.variables[append_dim]
    other_dim_points = np.atleast_1d(other_dim_var[:])

    # Check for the dataset being scalar on the append dimension.
    scalar_coord = False
    if len(other_dim_points) == 1:
        scalar_coord = True

    if scalar_coord:
        shape = [1] + list(data_var_shape)
    else:
        shape = data_var_shape

    offsets = []
    try:
        offset = _dim_offsets(
            other_dim_points, self_ind_stop, self_dim_stop, self_step,
            scalar=scalar_coord)
        offsets = [0] * len(shape)
        offsets[append_axis] = offset
        offset_inds = _array_indices(shape, offsets)
    except Exception as e:
        logging.info(f'{other_data_model.netcdf_filename} - {e}')

    # Append the data from other.
    data_array_path = os.path.join(domain_path, data_array_name)
    write_multiattr_array(data_array_path, other_data_vars, start_index=offset_inds)
    # Append the extra dimension points from other.
    dim_array_path = os.path.join(domain_path, append_dim)
    write_array(dim_array_path, other_dim_var, start_index=offset_inds[append_axis])


def _make_tile(other, domain_path, var_name, append_axis, append_dim,
               self_ind_stop, self_dim_stop, self_step,
               make_data_model, verbose, i=None, num=None):
    """Process appending a single tile to `self`."""
    if make_data_model:
        other_data_model = NCDataModel(other)
        other_data_model.classify_variables()
        other_data_model.get_metadata()
    else:
        other_data_model = other

    _progress_report(other_data_model, verbose, i, num)

    other_data_var = other_data_model.variables[var_name]
    other_dim_var = other_data_model.variables[append_dim]
    other_dim_points = np.atleast_1d(other_dim_var[:])

    # Check for the dataset being scalar on the append dimension.
    scalar_coord = False
    if len(other_dim_points) == 1:
        scalar_coord = True

    if scalar_coord:
        shape = [1] + list(other_data_var.shape)
    else:
        shape = other_data_var.shape

    offsets = []
    try:
        offset = _dim_offsets(
            other_dim_points, self_ind_stop, self_dim_stop, self_step,
            scalar=scalar_coord)
        offsets = [0] * len(shape)
        offsets[append_axis] = offset
        offset_inds = _array_indices(shape, offsets)
    except Exception as e:
        logging.info(f'{other_data_model.netcdf_filename} - {e}')

    # Append the data from other.
    data_array_path = os.path.join(domain_path, var_name)
    write_array(data_array_path, other_data_var, start_index=offset_inds)
    # Append the extra dimension points from other.
    dim_array_path = os.path.join(domain_path, append_dim)
    write_array(dim_array_path, other_dim_var, start_index=offset_inds[append_axis])
