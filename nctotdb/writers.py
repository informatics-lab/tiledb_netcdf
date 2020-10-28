from collections import defaultdict, namedtuple
import copy
import json
import logging
import os

import numpy as np
import tiledb
import zarr

from .data_model import NCDataModel
from .grid_mappings import store_grid_mapping
from .paths import PosixArrayPath, AzureArrayPath
from . import utils


append_arg_list = ['other', 'domain', 'name', 'axis', 'dim',
                   'ind_stop', 'dim_stop', 'step', 'scalar',
                   'mapping', 'verbose', 'job_number', 'n_jobs',
                   'make_data_model', 'offset', 'ctx', 'logfile']
defaults = [None] * len(append_arg_list)
AppendArgs = namedtuple('AppendArgs', append_arg_list, defaults=defaults)


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
            vec_interp = np.vectorize(fillnan)
            coord_points[missing_points] = vec_interp(ind_points[missing_points],
                                                      coord_points[0],
                                                      numeric_step)

            # Write the whole filled array back to the TileDB coord array.
            with tiledb.open(coord_array_path, 'w', ctx=self.ctx) as D:
                D[ned[0]:ned[1]] = coord_points
        else:
            if verbose:
                print(f'No missing points in {coord_array_name!r}, nothing to do.')


class _TDBWriter(Writer):
    """
    .. deprecated::
        This class is deprecated in favour of the former `MultiAttrTDBWriter`,
        now renamed to just `TDBWriter`. This preferred class provides all the
        functionality of this class but with extra functionality for writing
        multi-attr arrays as well.

        This class is being maintained for now as it provides some of the
        functionality the preferred class relies upon.

    Write Python objects loaded from NetCDF to TileDB.

    Data Model: an instance of `NCDataModel` supplying data from a NetCDF file.
    Filepath: the filepath to save the tiledb array at.

    """
    def __init__(self, data_model,
                 array_filepath=None, container=None, array_name=None,
                 unlimited_dims=None, ctx=None):
        super().__init__(data_model, array_filepath, container, array_name,
                         unlimited_dims, ctx)

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

        # Handle scalar dimensions.
        if dim_name == self._scalar_unlimited:
            dim_coord_len = 1
            chunks = (1,)
        else:
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

    def create_domain_arrays(self, domain_vars, domain_name, coords=False):
        """Create one single-attribute array per data var in this NC domain."""
        for var_name in domain_vars:
            # Set dims for the enclosing domain.

            data_var = self.data_model.variables[var_name]
            data_var_dims = data_var.dimensions
            # Handle scalar append dimension coordinates.
            if not len(data_var_dims) and var_name == self._scalar_unlimited:
                data_var_dims = [self._scalar_unlimited]
            array_dims = [self._create_tdb_dim(dim_name, coords) for dim_name in data_var_dims]
            tdb_domain = tiledb.Domain(*array_dims)

            # Get tdb attributes.
            attr = tiledb.Attr(name=var_name, dtype=data_var.dtype)

            # Create the URI for the array.
            array_filename = self.array_path.construct_path(domain_name, var_name)
            # Create an empty array.
            schema = tiledb.ArraySchema(domain=tdb_domain, sparse=False,
                                        attrs=[attr], ctx=self.ctx)
            tiledb.Array.create(array_filename, schema)

    def _array_indices(self, shape, start_index):
        """Set the array indices to write the array data into."""
        return _array_indices(shape, start_index)

    def _get_grid_mapping(self, data_var):
        """
        Get the data variable's grid mapping variable, encode it as a JSON string
        for easy storage in the TileDB array's meta and return it for storage as
        array metadata.

        """
        grid_mapping_name = data_var.getncattr("grid_mapping")
        result = 'none'  # TileDB probably won't support `NoneType` in array meta.
        if grid_mapping_name is not None:
            assert grid_mapping_name in self.data_model.grid_mapping
            grid_mapping_var = self.data_model.variables[grid_mapping_name]
            result = store_grid_mapping(grid_mapping_var)
        return result

    def populate_array(self, var_name, data_var, domain_name,
                       start_index=None, write_meta=True):
        """Write the contents of a netcdf data variable into a tiledb array."""
        # Get the data variable and the URI of the array to write to.
        var_name = data_var.name
        array_filename = self.array_path.construct_path(domain_name, var_name)

        # Write to the array.
        scalar = var_name == self._scalar_unlimited
        write_array(array_filename, data_var,
                    start_index=start_index, scalar=scalar, ctx=self.ctx)
        if write_meta:
            with tiledb.open(array_filename, 'w', ctx=self.ctx) as A:
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
                    # Add grid mapping metadata as a JSON string.
                    grid_mapping_string = self._get_grid_mapping(data_var)
                    A.meta['grid_mapping'] = grid_mapping_string
                    # XXX: can't add list or tuple as values to metadata dictionary...
                    dim_coord_names = self._get_dim_coord_names(var_name)
                    A.meta['dimensions'] = ','.join(n for n in dim_coord_names)
                elif var_name in self.data_model.dim_coord_names:
                    # A dim coord gets a `coord` key in the metadata dictionary,
                    # value being the name of the coordinate.
                    A.meta['coord'] = self.data_model.dimensions[var_name].name
                elif var_name == self._scalar_unlimited:
                    # Handle scalar coords along the append axis.
                    A.meta['coord'] = self._scalar_unlimited
                else:
                    # Don't know how to handle this. It might be an aux or scalar
                    # coord, but we're not currently writing TDB arrays for them.
                    pass

    def populate_domain_arrays(self, domain_vars, domain_name):
        """Populate all arrays with data from netcdf data vars within a tiledb group."""
        for var_name in domain_vars:
            data_var = self.data_model.variables[var_name]
            self.populate_array(var_name, data_var, domain_name)

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
            group_dirname = self.array_path.construct_path(domain_name, '')
            if self.array_filepath is not None:
                # XXX TileDB does not automatically create group directories.
                self._create_tdb_directory(group_dirname)
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
        domain_name = self._public_domain_name(domain)
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

    def fill_missing_points(self, append_dim_name, verbose=False):
        # XXX duplicated from `append` method.
        domain = self.data_model.varname_domain_mapping[var_name]
        domain_name = f'domain_{self.data_model.domains.index(domain)}'
        domain_path = os.path.join(self.array_filepath, self.array_name, domain_name)
        coord_array_path = os.path.join(domain_path, append_dim_name)

        self._fill_missing_points(coord_array_path, append_dim_name, verbose=verbose)


class TileDBWriter(_TDBWriter):
    """
    Write Python objects loaded from NetCDF to TileDB, including writing TileDB
    arrays with multiple data attributes from different NetCDF data variables
    that are equally dimensioned.

    Data Model: an instance of `NCDataModel` supplying data from a NetCDF file.
    Filepath: the filepath to save the tiledb array at.

    """
    def __init__(self, data_model,
                 array_filepath=None, container=None, array_name=None,
                 unlimited_dims=None, ctx=None, domain_separator=','):
        """
        Set up a writer to store the contents of one or more NetCDF data models
        in a TileDB array.

        Args:
        * data_model: An `NCDataModel` object. Defines the contents of the base NetCDF file
                      to write to TileDB.

        Kwargs:
        * array_filepath: posix-like path of location on disk to write the TileDB array.
                          Either `array_filepath` or `container` *must* be provided, but not both.
        * container: the name of an Azure Storage container to write the TileDB array to.
                     Either `array_filepath` or `container` *must* be provided, but not both.
        * array_name: the name of the root TileDB array to write. Defaults to the name of the
                      data model's NetCDF file if not set. By specifying `array_name` as a
                      relative path along with `container` you can write the TileDB array
                      to a location other than the Azure Storage container's root.
        * unlimited_dims: a named dimension that will have unlimited length in the written
                          TileDB array. Typically this is set to a dimension you wish to append to.
                          The string given here must be the name of a dimension in the supplied
                          `data_model`.
        * ctx: a TileDB Ctx (context) object, typically used to store metadata relating to the
               Azure Storage container.
        * domain_separator: a string that specifies the separator between dimensions in
                            domain names. Defaults to `,`.

        """
        super().__init__(data_model, array_filepath, container, array_name,
                         unlimited_dims, ctx)

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
            # Promote scalar append dimensions.
            if self.unlimited_dims not in dimensions and self.unlimited_dims in self.data_model.scalar_coord_names:
                dimensions = [self.unlimited_dims] + list(dimensions)
                self._scalar_unlimited = self.unlimited_dims
            domain_string = self._public_domain_name(dimensions, self.domain_separator)
            shape_domains.append((domain_string, data_var_name))

        domains_mapping = defaultdict(list)
        for domain, data_var_name in shape_domains:
            domains_mapping[domain].append(data_var_name)
        self.domains_mapping = domains_mapping

    def populate_multiattr_array(self, data_array_name, data_var_names, domain_name,
                                 start_index=None, write_meta=True):
        """Write the contents of multiple data variables into a multi-attr TileDB array."""
        array_filename = self.array_path.construct_path(domain_name, data_array_name)

        # Write to the array.
        data_vars = {name: self.data_model.variables[name] for name in data_var_names}
        scalar = self._scalar_unlimited is not None
        write_multiattr_array(array_filename, data_vars,
                              start_index=start_index, scalar=scalar, ctx=self.ctx)
        if write_meta:
            multi_attr_metadata = self._multi_attr_metadata(data_var_names)
            with tiledb.open(array_filename, 'w', ctx=self.ctx) as A:
                # Set tiledb metadata from data var ncattrs.
                for key, value in multi_attr_metadata.items():
                    A.meta[key] = value
                # A data array gets a `dataset` key in the metadata dictionary,
                # which defines all the data variables in it.
                A.meta['dataset'] = ','.join(data_var_names)
                # Define this as being a multi-attr array.
                A.meta['multiattr'] = True
                # Add grid mapping metadata as a JSON string.
                zeroth_data_var = list(data_vars.values())[0]
                grid_mapping_string = self._get_grid_mapping(zeroth_data_var)
                A.meta['grid_mapping'] = grid_mapping_string
                # XXX: can't add list or tuple as values to metadata dictionary...
                dim_coord_names = self._get_dim_coord_names(data_var_names[0])
                A.meta['dimensions'] = ','.join(n for n in dim_coord_names)

    def create_multiattr_array(self, domain_var_names, domain_dims,
                               domain_name, data_array_name):
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
        array_filename = self.array_path.construct_path(domain_name, data_array_name)
        # Create an empty array.
        schema = tiledb.ArraySchema(domain=tdb_domain, sparse=False,
                                    attrs=attrs, ctx=self.ctx)
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
            group_dirname = self.array_path.construct_path(domain_name, '')
            # XXX This might be failing because the TileDB root dir doesn't exist...
            # For a POSIX path we must explicitly create the group directory.
            if self.array_filepath is not None:
                # TODO why is this necessary? Shouldn't tiledb create if this dir does not exist?
                self._create_tdb_directory(group_dirname)

            tiledb.group_create(group_dirname, ctx=self.ctx)

            # Create and write arrays for each domain-describing coordinate.
            self.create_domain_arrays(domain_coord_names, domain_name, coords=True)
            self.populate_domain_arrays(domain_coord_names, domain_name)

            # Get data vars in this domain and create and populate a multi-attr array.
            self.create_multiattr_array(domain_var_names, domain_coord_names,
                                        domain_name, data_array_name)
            self.populate_multiattr_array(data_array_name, domain_var_names, domain_name)

    def _scalar_step(self, base_point, append_dim, other):
        """
        Manually calculate the append dimension point step in the scalar case
        when it cannot be done by finding the diff between successive points.

        """
        other_data_model = NCDataModel(other)
        other_data_model.classify_variables()
        offset_point = other_data_model.variables[append_dim][:]
        return offset_point - base_point

    def _get_scalar_points_and_offsets(self, others, append_dim, self_dim_stop):
        odp = []
        for other in others:
            ncdm = NCDataModel(other)
            with ncdm.open_netcdf():
                ncdm.classify_variables()
                ncdm.get_metadata()
                odp.append(ncdm.variables[append_dim][:])
        other_dim_points = np.array(odp)
        offsets = other_dim_points - self_dim_stop
        return offsets.data  # Only return the non-masked element of the masked array.

    def _run_consolidate(self, domain_names, data_array_name, verbose=False):
        # Consolidate at the end of the append operations to make the resultant
        # array more performant.
        config_key_name = "sm.consolidation.steps"
        config_key_value = 100
        if self.ctx is None:
            config = tiledb.Config({config_key_name: config_key_value})
            ctx = tiledb.Ctx(config)
        else:
            cfg_dict = self.ctx.config().dict()
            cfg_dict[config_key_name] = config_key_value
            ctx = tiledb.Ctx(config=tiledb.Config(cfg_dict))
        for i, domain_name in enumerate(domain_names):
            if verbose:
                print()  # Clear last carriage-returned print statement.
                print(f'Consolidating array: {i+1}/{len(domain_names)}', end="\r")
            else:
                print('Consolidating...')
            array_path = self.array_path.construct_path(domain_name, data_array_name)
            tiledb.consolidate(array_path, ctx=ctx)

    def append(self, others, append_dim, data_array_name,
               baseline=None, logfile=None, parallel=False,
               verbose=False, consolidate=True):
        """
        Append extra data as described by the contents of `others` onto
        an existing TileDB array along the axis defined by `append_dim`.

        Notes:
          * extends one dimension only
          * cannot create new dimensions, only extend existing dimensions

        TODO support multiple axis appends.
        TODO check if there's already data at the write inds and add an overwrite?

        """
        make_data_model = False
        # Check what sort of thing `others` is.
        if isinstance(others, (NCDataModel, str)):
            others = [others]

        # Check all domains for including the append dimension.
        domain_names = [d for d in self.domains_mapping.keys()
                        if append_dim in d.split(self.domain_separator)]
        domain_paths = [self.array_path.construct_path(d, '') for d in domain_names]
        append_axes = [d.split(self.domain_separator).index(append_dim) for d in domain_names]

        # Get starting dimension points and offsets.
        self_dim_var = self.data_model.variables[append_dim]
        self_dim_points = copy.copy(np.array(self_dim_var[:], ndmin=1))

        if append_dim == self._scalar_unlimited:
            if baseline is None:
                raise ValueError('Cannot determine scalar step without a baseline dataset.')
            self_ind_stop = 0
            self_dim_stop = self_dim_points[0]
            offsets = self._get_scalar_points_and_offsets(others, append_dim, self_dim_stop)
            if len(offsets) == 1:
                self_step = offsets[0]
            else:
                # Smooth out any noise in slightly different offsets.
                self_step = np.median(np.diff(offsets))
            scalar = True
        else:
            self_dim_start, self_dim_stop, self_step = _dim_points(self_dim_points)
            self_ind_start, self_ind_stop = _dim_inds(self_dim_points,
                                                      [self_dim_start, self_dim_stop])
            offsets = None
            scalar = False

        # For multidim / multi-attr appends this may be more complex.
        n_jobs = len(others)
        # Prepare for serialization.
        tdb_config = self.ctx.config().dict() if self.ctx is not None else None
        all_job_args = []
        for ct, other in enumerate(others):
            offset = offsets[ct] if offsets is not None else None
            this_job_args = AppendArgs(other=other, domain=domain_paths, name=data_array_name,
                                       dim=append_dim, axis=append_axes, scalar=scalar,
                                       offset=offset, mapping=self.domains_mapping, logfile=logfile,
                                       ind_stop=self_ind_stop, dim_stop=self_dim_stop, step=self_step,
                                       verbose=verbose, job_number=ct, n_jobs=n_jobs, ctx=tdb_config)
            all_job_args.append(this_job_args)

        # Serialize to JSON for network transmission.
        serialized_jobs = map(lambda job: json.dumps(job._asdict()), all_job_args)
        if parallel:
            import dask.bag as db
            bag_of_jobs = db.from_sequence(serialized_jobs)
            bag_of_jobs.map(_make_multiattr_tile_helper).compute()
        else:
            for job_args in serialized_jobs:
                _make_multiattr_tile_helper(job_args)

        if consolidate and (n_jobs > 10):
            self._run_consolidate(domain_names, data_array_name, verbose=verbose)

    def fill_missing_points(self, append_dim_name, verbose=False):
        # Check all domains for including the append dimension.
        coord_array_paths = []
        for domain_name in self.domains_mapping.keys():
            if append_dim_name in domain_name.split(self.domain_separator):
                coord_array_path = self.array_path.construct_path(domain_name,
                                                                  append_dim_name)
                self._fill_missing_points(coord_array_path,
                                          append_dim_name,
                                          verbose=verbose)


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


def write_array(array_filename, data_var,
                start_index=None, scalar=False, ctx=None):
    """Write to the array."""
    if start_index is None:
        start_index = 0
        if scalar:
            shape = (1,)
        else:
            shape = data_var.shape
        write_indices = _array_indices(shape, start_index)
    else:
        write_indices = start_index

    # Write netcdf data var contents into array.
    with tiledb.open(array_filename, 'w', ctx=ctx) as A:
        A[write_indices] = data_var[...]


def write_multiattr_array(array_filename, data_vars,
                          start_index=None, scalar=False, ctx=None):
    """Write to each attr in the array."""
    if start_index is None:
        start_index = 0
        zeroth_key = list(data_vars.keys())[0]
        shape = data_vars[zeroth_key].shape  # All data vars *must* have the same shape for writing...
        if scalar:
            shape = (1,) + shape
        write_indices = _array_indices(shape, start_index)
    else:
        write_indices = start_index

    # Write netcdf data var contents into array.
    with tiledb.open(array_filename, 'w', ctx=ctx) as A:
        A[write_indices] = {name: data_var[...] for name, data_var in data_vars.items()}


def _dim_inds(dim_points, spatial_inds, offset=0):
    """Convert coordinate values to index space."""
    return [list(dim_points).index(si) + offset for si in spatial_inds]


def _dim_points(points):
    """Convert a dimension variable (coordinate) points to index space."""
    start, stop = points[0], points[-1]
    step, = np.unique(np.diff(points))
    return start, stop, step


def _dim_offsets(dim_points, self_stop_ind, self_stop, self_step,
                 scalar=False, points_offset=None):
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

    if points_offset is None:
        points_offset = other_start - self_stop
    inds_offset = int(points_offset / self_step) + self_stop_ind

    i_start, _ = _dim_inds(dim_points, spatial_inds, inds_offset)
    return i_start


def fillnan(xi, y0, diff):
    """A simple linear 1D interpolator."""
    return y0 + (xi * diff)


def _progress_report(other_data_model, verbose, i, total):
    """A helpful printout of append progress."""
    # XXX Not sure about this when called from a bag...
    if verbose and i is not None and total is not None:
        fn = other_data_model.netcdf_filename
        ct = i + 1
        pc = 100 * (ct / total)
        print(f'Processing {fn}...  ({ct}/{total}, {pc:0.1f}%)', end="\r")


def _make_multiattr_tile(other_data_model, domain_path, data_array_name,
                         var_names, append_axis, append_dim, scalar_coord,
                         self_ind_stop, self_dim_stop, self_step,
                         scalar_offset=None, do_logging=False, ctx=None):
    """Process appending a single tile to `self`, per domain."""
    other_data_vars = {var_name: other_data_model.variables[var_name] for var_name in var_names}
    data_var_shape  = other_data_vars[0].shape
    other_dim_var = other_data_model.variables[append_dim]
    other_dim_points = np.atleast_1d(other_dim_var[:])

    # Check for the dataset being scalar on the append dimension.
    if not scalar_coord and len(other_dim_points) == 1:
        scalar_coord = True

    if scalar_coord:
        shape = [1] + list(data_var_shape)
    else:
        shape = data_var_shape

    offsets = []
    offset = _dim_offsets(
        other_dim_points, self_ind_stop, self_dim_stop, self_step,
        scalar=scalar_coord, points_offset=scalar_offset)
    offsets = [0] * len(shape)
    offsets[append_axis] = offset
    offset_inds = _array_indices(shape, offsets)
    domain_name = domain_path.split('/')[-1]
    if do_logging:
        logging.error(f'Indices for {other_data_model.netcdf_filename!r} ({domain_name}): {offset_inds}')

    # Append the data from other.
    data_array_path = f"{domain_path}{data_array_name}"
    write_multiattr_array(data_array_path, other_data_vars,
                          start_index=offset_inds, ctx=ctx)
    # Append the extra dimension points from other.
    dim_array_path = f"{domain_path}{append_dim}"
    write_array(dim_array_path, other_dim_var,
                start_index=offset_inds[append_axis], ctx=ctx)

    # I think this got added spuriously...
    # dim_array_path = f"{domain_path}{append_dim}"
    # write_array(dim_array_path, other_dim_var,
    #             start_index=offset_inds[append_axis], ctx=ctx)


def _make_multiattr_tile_helper(serialized_job):
    """
    Helper function to collate the processing of each file in a multi-attr append.

    """
    # Deserialize job args.
    job_args = AppendArgs(**json.loads(serialized_job))
    if job_args.ctx is not None:
        ctx = tiledb.Ctx(config=tiledb.Config(job_args.ctx))
    else:
        ctx = None

    do_logging = False
    if job_args.logfile is not None:
        do_logging = True
        logging.basicConfig(filename=job_args.logfile,
                            level=logging.ERROR,
                            format='%(asctime)s %(message)s',
                            datefmt='%d/%m/%Y %H:%M:%S')

    domains_mapping = job_args.mapping
    domain_paths = job_args.domain
    append_dim = job_args.dim
    append_axes = job_args.axis

    # Record what we've processed...
    if do_logging:
        logging.error(f'Processing {job_args.other!r} ({job_args.job_number+1}/{job_args.n_jobs})')

    # To improve fault tolerance all the append processing happens in a try/except...
    try:
        if isinstance(job_args.other, NCDataModel):
            other_data_model = job_args.other
        else:
            other_data_model = NCDataModel(job_args.other)

        with other_data_model.open_netcdf():
            for n, domain_path in enumerate(domain_paths):
                if job_args.verbose:
                    fn = other_data_model.netcdf_filename
                    job_no = job_args.job_number
                    n_jobs = job_args.n_jobs
                    n_domains = len(domain_paths)
                    print(f'Processing {fn}...  ({job_no+1}/{n_jobs}, domain {n+1}/{n_domains})', end="\r")

                append_axis = append_axes[n]
                if domain_path.endswith('/'):
                    _, domain_name = os.path.split(domain_path[:-1])
                else:
                    _, domain_name = os.path.split(domain_path)
                array_var_names = domains_mapping[domain_name]
                _make_multiattr_tile(other_data_model, domain_path, job_args.name,
                                     array_var_names, append_axis, append_dim, job_args.scalar,
                                     job_args.ind_stop, job_args.dim_stop, job_args.step,
                                     scalar_offset=job_args.offset, do_logging=do_logging, ctx=ctx)
    except Exception as e:
        emsg = f'Could not process {job_args.other!r}. Details:\n{e}\n'
        logging.error(emsg, exc_info=True)
        if job_args.logfile is None and job_args.verbose:
            raise

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


def _make_tile_helper(args):
    """Helper method to call from a `map` operation and unpack the args."""
    _make_tile(*args)
