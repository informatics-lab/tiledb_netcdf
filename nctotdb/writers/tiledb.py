from collections import defaultdict, namedtuple
import copy
import json
import logging
import os

import numpy as np
import tiledb

from .core import Writer
from ..data_model import NCDataModel, NCDataModelGroup
from ..grid_mappings import store_grid_mapping


append_arg_list = ['other', 'name', 'offsets', 'axes',
                   'mapping', 'verbose', 'job_number', 'n_jobs',
                   'group','ctx', 'logfile']
defaults = [None] * len(append_arg_list)
AppendArgs = namedtuple('AppendArgs', append_arg_list, defaults=defaults)


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
        result = 'none'  # TileDB probably won't support `NoneType` in array meta.
        try:
            grid_mapping_name = data_var.getncattr("grid_mapping")
        except AttributeError:
            pass
        else:
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
        domain = self.data_model.varname_domain_mapping[append_dim_name]
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
        ncattrs = data_var.ncattrs()
        metadata = {}
        for ncattr in ncattrs:
            metadata[ncattr] = data_var.getncattr(ncattr)
        # Add a fallback long_name in case of both standard_name and long_name being missing.
        if "standard_name" not in ncattrs and "long_name" not in ncattrs:
            metadata["long_name"] = data_var.name
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

    def create_domains(self, data_array_name='data', domains_mapping=None):
        """
        Create one TileDB domain for each unique shape / dimensions combination
        in the input Data Model. Each domain will contain:
          * one multi-attr array, where the attrs are all the data variables described
            by this combination of dimensions, and
          * one array for each of the dimension-describing coordinates for this
            combination of dimensions.

        """
        self._make_shape_domains()
        if domains_mapping is None:
            domains_mapping = self.domains_mapping

        for domain_name, domain_var_names in domains_mapping.items():
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

    def _get_scalar_offset(self, baseline, append_dim, self_dim_stop):
        """
        Use the specified baseline file to calcuate the single offset between every
        successive step along the scalar append dimension.

        """
        odm = NCDataModel(baseline)
        with odm.open_netcdf():
            odm.classify_variables()
            odm.get_metadata()
            points = np.atleast_1d(odm.variables[append_dim][:])
        result = points - self_dim_stop
        return result[0]

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

    def _baseline_append_offsets(self, append_dim_name, baseline=None, override_offset=None):
        """
        Determine the length of the base array (represented by `self`) along a
        named append dimension `append_dim_name`. Return the following values,
        which can be used to calculate the offset along the append dimension of
        a given chunk of data to be appended:
          * base_ind_stop: _index_ of last value along the named append dimension
                           in the base array (`self`)
          * base_dim_stop: last _dimension_ (coord) value along the named append
                           dimension in the base array (`self`)
          * dim_step: the difference between two successive _dimension_ (coord) values,
                      assumed to be constant throughout the dimension
          * scalar: whether the append dimension is made up of single (scalar) values

        """
        # Get starting dimension points and offsets.
        base_dim_var = self.data_model.variables[append_dim_name]
        base_dim_points = copy.copy(np.array(base_dim_var[:], ndmin=1))
        if append_dim_name == self._scalar_unlimited:
            if baseline is None:
                raise ValueError('Cannot determine scalar step without a baseline dataset.')
            base_ind_stop = 0
            base_dim_stop = base_dim_points[0]
            dim_step = self._get_scalar_offset(baseline, append_dim_name, base_dim_stop)
            scalar = True
        else:
            base_dim_start, base_dim_stop, dim_step = _dim_points(base_dim_points)
            _, base_ind_stop = _dim_inds(base_dim_points, [base_dim_start, base_dim_stop])
            scalar = False
        if override_offset is not None:
            dim_step = override_offset
        return int(base_ind_stop), int(base_dim_stop), int(dim_step), scalar

    def append(self, others, append_dims, data_array_name,
               baselines=None, override_offsets=None, group=False,
               logfile=None, parallel=False, verbose=False, consolidate=True):
        """
        Append extra data as described by the contents of `others` onto
        an existing TileDB array along the axis defined by `append_dim`.

        Notes:
          * extends one dimension only
          * cannot create new dimensions, only extend existing dimensions
          * `others` must be str of one or more files to append, not NCDataModel

        TODO support multiple axis appends.
        TODO check if there's already data at the write inds and add an overwrite?

        """
        # Check what sort of thing `others` is.
        if isinstance(others, str):
            others = [others]

        if isinstance(append_dims, str):
            append_dims = [append_dims]

        # Check all domains for including the append dimension.
        append_dims_set = set(append_dims)
        domain_names = []
        for d in self.domains_mapping.keys():
            domain_dims = d.split(self.domain_separator)
            if not len(append_dims_set - set(domain_dims)):
                domain_names.append(d)

        # Determine axes covered by each domain.
        domain_axes = {}
        for name in domain_names:
            domain_path = self.array_path.construct_path(name, '')
            axis_names = name.split(self.domain_separator)
            domain_axes[domain_path] = axis_names

        # Calculate base offsets along each append dimension.
        append_offsets = {}
        for append_dim in append_dims:
            baseline = override_offset = None
            if baselines is not None:
                baseline = baselines.get(append_dim, None)
            if override_offsets is not None:
                override_offset = override_offsets.get(append_dim, None)
            append_offsets[append_dim] = self._baseline_append_offsets(
                append_dim, baseline=baseline, override_offset=override_offset)

        # Set up logging.
        if logfile is not None:
            logging.basicConfig(filename=logfile,
                                level=logging.ERROR,
                                format='%(asctime)s %(message)s',
                                datefmt='%d/%m/%Y %H:%M:%S')

        n_jobs = len(others)
        # Prepare for serialization.
        tdb_config = self.ctx.config().dict() if self.ctx is not None else None
        all_job_args = []
        for ct, other in enumerate(others):
            this_job_args = AppendArgs(other=other, name=data_array_name,
                                       offsets=append_offsets, axes=domain_axes, group=group,
                                       mapping=self.domains_mapping, logfile=logfile,
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

# #####
# Static helper functions.
# #####


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
    # Determine shape of items to be written.
    zeroth_key = list(data_vars.keys())[0]
    shape = data_vars[zeroth_key].shape  # All data vars *must* have the same shape for writing...
    if scalar:
        shape = (1,) + shape

    # Get write indices.
    if start_index is None:
        start_index = 0
        write_indices = _array_indices(shape, start_index)
    else:
        write_indices = start_index

    # Check for attrs with no data.
    for name, data_var in data_vars.items():
        if data_var is None:
            # Handle missing data for this attr.
            missing_data = np.empty(shape)
            missing_data.fill(np.nan)
            data_vars[name] = missing_data

    # Write netcdf data var contents into array.
    with tiledb.open(array_filename, 'w', ctx=ctx) as A:
        A[write_indices] = {name: data_var[...] for name, data_var in data_vars.items()}


def _dim_inds(dim_points, spatial_inds, offset=0):
    """Convert coordinate values to index space."""
    return [list(dim_points).index(si) + offset for si in spatial_inds]


def _dim_points(points):
    """Convert a dimension variable (coordinate) points to index space."""
    points_ndim = points.ndim
    if points_ndim != 1:
        emsg = f"The append dimension must be 1D, got {points_ndim}D array."
        raise ValueError(emsg)
    start, stop = points[0], points[-1]
    step, = np.unique(np.diff(points))
    return start, stop, step


def _dim_offsets(dim_points, self_stop_ind, self_stop, self_step,
                 scalar=False):
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
    if verbose and i is not None and total is not None:
        fn = other_data_model.netcdf_filename
        ct = i + 1
        pc = 100 * (ct / total)
        print(f'Processing {fn}...  ({ct}/{total}, {pc:0.1f}%)', end="\r")


def _make_multiattr_tile(other_data_model, domain_path, data_array_name,
                         var_names, domain_axes, append_offsets,
                         do_logging=False, ctx=None):
    """Process appending a single tile to `self`, per domain."""
    other_data_vars = {}
    for maybe_hashed_name in var_names:
        try:
            other_data_vars[maybe_hashed_name] = other_data_model.variables[maybe_hashed_name]
        except KeyError:
            raise ValueError(f"No data var {maybe_hashed_name!r}!")

    # Raise an error if no match in data vars between existing array and other_data_model.
    if len(list(other_data_vars.keys())) == 0:
        emsg = "Variable names in data model [{}] not present in existing array."
        raise KeyError(emsg.format(', '.join(other_data_model.data_var_names)))

    zeroth_data_var = list(other_data_vars.keys())[0]
    data_var_shape  = other_data_vars[zeroth_data_var].shape

    offsets = []
    append_axes = domain_axes[domain_path]
    for append_dim, (ind_stop, dim_stop, step, scalar_coord) in append_offsets.items():
        other_dim_var = other_data_model.variables[append_dim]
        other_dim_points = np.atleast_1d(other_dim_var[:])
        append_axis = append_axes.index(append_dim)

        # Check for the dataset being scalar on the append dimension.
        if not scalar_coord and len(other_dim_points) == 1:
            scalar_coord = True

        shape = [1] + list(data_var_shape) if scalar_coord else data_var_shape

        offset = _dim_offsets(
            other_dim_points, ind_stop, dim_stop, step,
            scalar=scalar_coord)
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

    do_logging = job_args.logfile is not None

    domains_mapping = job_args.mapping
    domain_axes = job_args.axes
    domain_paths = job_args.axes.keys()
    append_offsets = job_args.offsets
    group = job_args.group

    # Record what we've processed...
    if do_logging:
        logging.error(f'Processing {job_args.other!r} ({job_args.job_number+1}/{job_args.n_jobs})')

    # To improve fault tolerance all the append processing happens in a try/except...
    try:
        if group:
            other_data_model = NCDataModelGroup(job_args.other)
        elif isinstance(job_args.other, NCDataModel):
            other_data_model = job_args.other
        else:
            other_data_model = NCDataModel(job_args.other)
            other_data_model.populate()

        with other_data_model.open_netcdf():
            for n, domain_path in enumerate(domain_paths):
                if job_args.verbose:
                    fn = other_data_model.netcdf_filename
                    job_no = job_args.job_number
                    n_jobs = job_args.n_jobs
                    n_domains = len(domain_paths)
                    print(f'Processing {fn}...  ({job_no+1}/{n_jobs}, domain {n+1}/{n_domains})', end="\r")

                if domain_path.endswith('/'):
                    _, domain_name = os.path.split(domain_path[:-1])
                else:
                    _, domain_name = os.path.split(domain_path)
                array_var_names = domains_mapping[domain_name]
                _make_multiattr_tile(other_data_model, domain_path, job_args.name,
                                     array_var_names, domain_axes, append_offsets,
                                     do_logging=do_logging, ctx=ctx)
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
