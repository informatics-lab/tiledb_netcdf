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
        with tiledb.open(array_filename, 'w') as A:
            # Write netcdf data var contents into array.
            if start_index is None:
                start_index = 0
                shape = data_var.shape
                write_indices = self._array_indices(shape, start_index)
            else:
                write_indices = start_index
            A[write_indices] = data_var[...]

            if write_meta:
                # Set tiledb metadata from data var ncattrs.
                for ncattr in data_var.ncattrs():
                    A.meta[ncattr] = data_var.getncattr(ncattr)
                # Add metadata describing whether this is a coord or data var.
                if var_name in self.data_model.data_var_names:
                    # A data var gets a `data_var` key in the metadata dictionary,
                    # value being all the dim coords that describe it.
                    A.meta['dataset'] = var_name
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

    def append(self, other_data_model, var_name, append_dim, offsets=None):
        """
        Append the data from a data variable in `other_data_model`
        by extending one dimension of that data variable in the tiledb
        described by `self`.

        Notes:
          * extends one dimension only on a single data variable
          * cannot create new dimensions, only extend existing dimensions

        Assumptions:
          * for now, that the data in other directly follows on from the
            data in self, so that there are no gaps or overlaps in the
            appended data

        """
        # Check if the append can go ahead.
        self._append_checker(other_data_model, var_name, append_dim)

        # Get data vars from self and other.
        self_data_var = self.data_model.variables[var_name]
        other_data_var = other_data_model.variables[var_name]

        # Get domain for var_name and tiledb array path.
        domain = self.data_model.varname_domain_mapping[var_name]
        domain_name = self._public_domain_name(domain)
        domain_path = os.path.join(self.array_filepath, self.array_name, domain_name)

        # Get the index and name of the append dimension.
        append_axis, append_dim = self._append_dimension(var_name, append_dim)
        other_dim_var = other_data_model.variables[append_dim]

        # Get the offset along the append dimension, assuming that self and other are
        # contiguous along this dimension.
        if offsets is None:
            with tiledb.open(os.path.join(domain_path, var_name), 'r') as A:
                array_shape = A.nonempty_domain()
            # We want to get the next index from the upper bound of
            # the nonempty domain along the append axis.
            append_dim_offset = array_shape[append_axis][1] + 1
            offsets = [0] * len(self_data_var.shape)
            offsets[append_axis] = append_dim_offset

        # Append the data from other.
        self.populate_array(var_name, other_data_var, domain_path,
                            start_index=offsets, write_meta=False)
        # Append the extra dimension points from other.
        self.populate_array(append_dim, other_dim_var, domain_path,
                            start_index=offsets[append_axis], write_meta=False)


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


# Remove these functions from `TDBWriter` because most of them are static and it
# might make parallel tiles possible!

def _array_indices(shape, start_index):
    """Set the array indices to write the array data into."""
    if isinstance(start_index, int):
        start_index = [start_index] * len(shape)

    array_indices = []
    for dim_len, start_ind in zip(shape, start_index):
        array_indices.append(slice(start_ind, dim_len+start_ind))
    return tuple(array_indices)


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


def _make_tile_helper(args):
    """Helper method to call from a `map` operation and unpack the args."""
    _make_tile(*args)


def _make_tile(writer, other, var_name, append_axis, append_dim,
               self_ind_stop, self_dim_stop, self_step,
               make_data_model, verbose, i=None, num=None):
    """Process appending a single tile to `self`."""
    if make_data_model:
        other_data_model = NCDataModel(other)
        other_data_model.classify_variables()
        other_data_model.get_metadata()
    else:
        other_data_model = other

    # XXX Not sure about this when called from a bag...
    if verbose and i is not None and verbose is not None:
        fn = other_data_model.netcdf_filename
        ct = i + 1
        pc = 100 * (ct / num)
        print(f'Processing {fn}...  ({ct}/{num}, {pc:0.1f}%)', end="\r")

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
            other_dim_points, self_ind_stop, self_dim_stop, self_step, scalar=scalar_coord)
        offsets = [0] * len(shape)
        offsets[append_axis] = offset
        # XXX think about this one!
        offset_inds = self._array_indices(shape, offsets)
        writer.append(other_data_model, var_name, append_dim, offsets=offset_inds)
    except Exception as e:
        logging.info(f'{other_data_model.netcdf_filename} - {e}')


def tile(writer, others, var_name, append_dim,
         logfile=None, parallel=False, verbose=False):
    """
    Enable multiple, possibly non-contiguous, eventually multi-axis
    append operations from multiple data model objects. This is done by
    working out where each tile fits relative to `self` in
    each append dimension.

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

    append_axis, append_dim = writer._append_dimension(var_name, append_dim)

    self_data_var = writer.data_model.variables[var_name]
    self_dim_var = writer.data_model.variables[append_dim]
    self_dim_points = self_dim_var[:]
    self_shape = self_data_var.shape
    self_dim_start, self_dim_stop, self_step = _dim_points(self_dim_points)
    self_ind_start, self_ind_stop = _dim_inds(self_dim_points,
                                              [self_dim_start, self_dim_stop])

    # For multidim / multi-attr appends this will be more complex.
    jobs = others
    common_job_args = [var_name, append_axis, append_dim,
                        self_ind_stop, self_dim_stop, self_step,
                        make_data_model, verbose]
    job_args = [[other] + common_job_args for other in others]

    if parallel:
        bag_of_jobs = db.from_sequence(job_args)
        bag_of_jobs.map(_make_tile_helper).compute()
    else:
        for i, args in enumerate(job_args):
            args += [i, len(job_args)]
            _make_tile_helper(args)