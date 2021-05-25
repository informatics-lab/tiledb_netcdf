from itertools import chain
import os
import warnings

import cf_units
import dask.array as da
from iris.coords import CellMethod, DimCoord
from iris.cube import Cube, CubeList
from iris.fileformats.netcdf import parse_cell_methods
import numpy as np
import tiledb

from .core import Reader
from ..grid_mappings import GridMapping
from ..proxy import TileDBDataProxy
from .. import utils


# Ref Iris: https://github.com/SciTools/iris/blob/master/lib/iris/_cube_coord_common.py#L75
IRIS_FORBIDDEN_KEYS = set([
        "standard_name",
        "long_name",
        "units",
        "bounds",
        "axis",
        "calendar",
        "leap_month",
        "leap_year",
        "month_lengths",
        "coordinates",
        "grid_mapping",
        "climatology",
        "cell_methods",
        "formula_terms",
        "compress",
        "add_offset",
        "scale_factor",
        "_FillValue",
    ])


class TileDBReader(Reader):
    def __init__(self, array_name, array_filepath=None, container=None,
                 storage_options=None, data_array_name=None, ctx=None):
        super().__init__(array_filepath)

        self.array_name = array_name
        self.container = container
        self.storage_options = storage_options
        self.data_array_name = data_array_name
        self.ctx = ctx

        self.groups = {}
        self._arrays = None

        # Need either a local filepath or a remote container.
        utils.ensure_filepath_or_container(self.array_filepath, self.container)
        self.array_path = utils.filepath_generator(self.array_filepath,
                                                   self.container,
                                                   self.array_name,
                                                   ctx=self.ctx)

    @property
    def arrays(self):
        if self._arrays is None:
            self._array_paths()
        return self._arrays

    @arrays.setter
    def arrays(self, value):
        self._arrays = value

    def check_groups(self):
        if not len(self.groups.keys()):
            self.get_groups_and_arrays()

    def _get_array_attrs(self, array_path):
        with tiledb.open(array_path, "r", ctx=self.ctx) as A:
            nattr = A.schema.nattr
            attr_names = [A.schema.attr(i).name for i in range(nattr)]
        return attr_names

    def _array_paths(self):
        """Produce a mapping of array name to array path irrespective of groups."""
        self.check_groups()
        all_paths = chain.from_iterable([paths for paths in self.groups.values()])
        arrays = {}
        for path in all_paths:
            if path.endswith('/'):
                _, array_name = os.path.split(path[:-1])
            else:
                _, array_name = os.path.split(path)
            if array_name == self.data_array_name:
                attr_names = self._get_array_attrs(path)
                for attr_name in attr_names:
                    arrays[attr_name] = path
            else:
                # XXX assumes that we will not have duplicated array names.
                arrays[array_name] = path
        self.arrays = arrays

    def classifier(self, item_path, item_type):
        """
        Store and classify items in a `tiledb.walk` operation as either a TileDB group
        or a TileDB array in a given TileDB group.

        """
        if item_type == 'group':
            self.groups[item_path] = []
        else:
            if item_path.endswith('/'):
                group, _ = os.path.split(item_path[:-1])
            else:
                group, _ = os.path.split(item_path)
            self.groups[group].append(item_path)

    def get_groups_and_arrays(self):
        tiledb.walk(self.array_path.basename, self.classifier, ctx=self.ctx)

    def tdb_dir_contents(self, dir):
        contents = []
        tiledb.ls(self.array_path.basename, lambda obj_path, _: contents.append(obj_path),
                  ctx=self.ctx)
        return contents

    def _get_dim_coords(self, array_filepath):
        """Get the dimension describing coordinates from a TileDB array."""
        with tiledb.open(array_filepath, 'r', ctx=self.ctx) as A:
            dims_string = A.meta['dimensions']
        return dims_string.split(',')

    def _map_coords_inds(self, group_name, dim_name, spatial_inds):
        """
        Map coordinate point values for a named dimension to indices in that
        dimension and return as a slice of indices.

        TODO handle time coordinates (by loading the calendar from the array meta?).

        """
        dim_filepath = self.array_path.construct_path(group_name, dim_name)
        with tiledb.open(dim_filepath, 'r', ctx=self.ctx) as D:
            coord_points = D[:]

        # XXX won't handle repeated values (which should never appear in a dim coord).
        index_inds = [coord_points.index(si) for si in spatial_inds]

        n_inds = len(index_inds)
        if n_inds == 1:
            result_slice = slice(index_inds, index_inds+1)
        else:
            max_inds = 3  # [start, stop, step]
            if n_inds < max_inds:
                index_inds += [None] * (max_inds-n_inds)
            result_slice = slice(*index_inds)
        return result_slice

    def spatial_index(self, group_name, array_name, spatial_inds):
        """
        Index a specified array in coordinate space rather than in index space.
        TileDB arrays are all described in index space, with named `Dim`s
        describing a `Domain` that encapsulates the array. Earth system data, however,
        typically is described as labelled arrays, with a named coordinate describing
        each dimension of the array.

        Practically, this provides a mapping from spatial indices (the input) to
        index space, which is used to index the array.

        NOTE: only spatial coordinate *values* are supported; datetimes in particular
        are not currently supported.

        """
        array_filepath = self.array_path.construct_path(group_name, array_name)
        array_dims = self._get_dim_coords(array_filepath)

        # Check that all the coords being spatially indexed are in the array's coords.
        coord_names = list(spatial_inds.keys())
        assert list(set(coord_names) & set(array_dims)) == coord_names

        indices = []
        for dim_name in array_dims:
            coord_vals = spatial_inds.get(dim_name, None)
            if coord_vals is None:
                indices.append(slice(None))
            else:
                dim_slice = self._map_coords_inds(group_name, dim_name, coord_vals)
                indices.append(dim_slice)

        with tiledb.open(array_filepath, 'r', ctx=self.ctx) as A:
            subarray = A[tuple(indices)]
        return subarray

    def _extract(self, array_name):
        """
        Return the path to a named array, plus paths for all the associated
        dimension arrays.

        Handles multi-attr arrays by scanning all attrs in arrays that match the data
        array name passed to `self` at instantiation.

        """
        # Sanity check the requested array name is in this TileDB.
        assert array_name in self.arrays.keys()
        named_array_path = self.arrays[array_name]

        named_group_path, _ = os.path.split(named_array_path)
        named_group_arrays = self.groups[named_group_path]

        with tiledb.open(named_array_path, 'r', ctx=self.ctx) as A:
            dim_names = A.meta['dimensions'].split(',')

        dim_paths = []
        for dim_name in dim_names:
            for array_path in named_group_arrays:
                array_path = array_path[:-1] if array_path.endswith('/') else array_path
                if array_path.endswith(dim_name):
                    dim_paths.append(array_path)
                    break
        # Confirm we have an array path for each dim_name.
        assert len(dim_paths) == len(dim_names)

        return named_array_path, dim_paths

    def _array_shape(self, nonempty_domain, slices=False):
        """
        Use the TileDB array's nonempty domain (i.e. the domain that encapsulates
        all written cells) to set the shape of the data to be read out of the
        TileDB array.

        """
        # We need to include the stop index, not exclude it.
        if slices:
            slices = [slice(start, stop+1, 1) for (start, stop) in nonempty_domain]
            return tuple(slices)  # Can only index with tuple, not list.
        else:
            # TileDB describes the nonempty domain quite annoyingly!
            return [filled[1]+1 for filled in nonempty_domain]

    def _handle_attributes(self, attrs, exclude_keys=None):
        """
        Iris contains a list of attributes that may not be written to a cube/coord's
        attributes dictionary. If any of thes attributes are present in a
        TileDB array's `meta`, remove them.

        Optionally also remove extra spurious keys defined with `exclude_keys` - such as
        dictionary items set by the writer (including `dataset` and `dimensions`.)

        """
        attrs_keys = set(attrs.keys())
        if exclude_keys is not None:
            allowed_keys = list(attrs_keys - IRIS_FORBIDDEN_KEYS - set(exclude_keys))
        else:
            allowed_keys = list(attrs_keys - IRIS_FORBIDDEN_KEYS)
        return {k: attrs[k] for k in allowed_keys}

    def _from_tdb_array(self, array_path, naming_key,
                        array_name=None, to_dask=False, handle_nan=None):
        """Retrieve data and metadata from a specified TileDB array."""
        with tiledb.open(array_path, 'r', ctx=self.ctx) as A:
            metadata = {k: v for k, v in A.meta.items()}
            if array_name is None:
                array_name = metadata[naming_key]
            if to_dask:
                schema = A.schema
                dtype = schema.attr(array_name).dtype
                chunks = [schema.domain.dim(i).tile for i in range(schema.ndim)]
                array_shape = self._array_shape(A.nonempty_domain())
                proxy = TileDBDataProxy(array_shape, dtype, array_path, array_name,
                                        handle_nan=handle_nan, ctx=self.ctx)
                points = da.from_array(proxy, chunks, name=naming_key)
            else:
                array_inds = self._array_shape(A.nonempty_domain(), slices=True)
                points = A[array_inds][array_name]
        return metadata, points

    def _load_dim(self, dim_path, grid_mapping):
        """
        Create an Iris DimCoord from a TileDB array describing a dimension.

        # TODO not handled here: circular.

        """
        metadata, points = self._from_tdb_array(dim_path, 'coord')

        coord_name = metadata.pop('coord')
        standard_name = metadata.pop('standard_name', None)
        long_name = metadata.pop('long_name', None)
        var_name = metadata.pop('var_name', None)

        # Check if we've a known horizontal coord name in order to write the
        # grid mapping as it's coord system.
        if standard_name in self.horizontal_coord_names:
            coord_system = grid_mapping
        else:
            coord_system = None

        units = metadata.pop('units')
        if coord_name == 'time':
            # Handle special-case complicated time coords.
            calendar = metadata.pop('calendar')
            units = cf_units.Unit(units, calendar=calendar)

        safe_attrs = self._handle_attributes(metadata)
        coord = DimCoord(points,
                         standard_name=standard_name,
                         long_name=long_name,
                         units=units,
                         attributes=safe_attrs,
                         coord_system=coord_system)
        return coord_name, coord

    def _load_group_dims(self, group_dim_paths, grid_mapping):
        """Load all dimension-describing (coordinate) arrays in the group."""
        group_dims = {}
        for dim_path in group_dim_paths:
            name, coord = self._load_dim(dim_path, grid_mapping)
            group_dims[name] = coord
        return group_dims

    def _load_data(self, array_path, group_dims, grid_mapping,
                   attr_name=None, separator='__', handle_nan=None):
        """
        Create an Iris cube from a TileDB array describing a data variable and
        pre-loaded dimension-describing coordinates.

        TODO not handled here: aux coords and dims, cell measures, aux factories.

        """
        single_attr_name = 'dataset'
        if attr_name is None:
            attr_metadata, lazy_data = self._from_tdb_array(array_path, single_attr_name,
                                                            to_dask=True, handle_nan=handle_nan)
            metadata = attr_metadata
            attr_name = metadata.pop(single_attr_name)
        else:
            attr_metadata, lazy_data = self._from_tdb_array(array_path,
                                                            single_attr_name,
                                                            array_name=attr_name,
                                                            to_dask=True,
                                                            handle_nan=handle_nan)
            metadata = {}
            for key, value in attr_metadata.items():
                # Varname-specific keys are of form `keyname__attrname`; we only want `keyname`.
                # TODO pass the separator character to the method.
                try:
                    key_name, key_attr = key.split(separator)
                    if key_attr == attr_name:
                        metadata[key_name] = value
                except ValueError:
                    # Not all keys are varname-specific; we want all of these.
                    metadata[key] = value

        cell_methods = parse_cell_methods(metadata.pop('cell_methods', None))
        dim_names = metadata.pop('dimensions').split(',')
        # Dim Coords And Dims (mapping of coords to cube axes).
        dcad = [(group_dims[name], i) for i, name in enumerate(dim_names)]
        safe_attrs = self._handle_attributes(metadata,
                                             exclude_keys=['dataset', 'multiattr', 'grid_mapping'])
        std_name = metadata.pop('standard_name', None)
        long_name = metadata.pop('long_name', None)
        var_name = metadata.pop('var_name', None)
        if all(itm is None for itm in [std_name, long_name, var_name]):
            long_name = attr_name

        cube = Cube(lazy_data,
                    standard_name=std_name,
                    long_name=long_name,
                    var_name=var_name,
                    units=metadata.pop('units', '1'),
                    dim_coords_and_dims=dcad,
                    cell_methods=cell_methods,
                    attributes=safe_attrs)
        cube.coord_system = grid_mapping
        return cube

    def _load_group_arrays(self, data_paths, group_dims, grid_mapping, handle_nan=None):
        """Load all data-describing (cube) arrays in the group."""
        cubes = []
        for data_path in data_paths:
            cube = self._load_data(data_path, group_dims, grid_mapping, handle_nan=handle_nan)
            cubes.append(cube)
        return cubes

    def _load_multiattr_arrays(self, data_paths, group_dims, grid_mapping,
                               attr_names=None, handle_nan=None):
        """Load all data-describing (cube) attrs from a multi-attr array."""
        if isinstance(attr_names, str):
            attr_names = [attr_names]

        cubes = []
        for data_path in data_paths:
            if attr_names is None:
                with tiledb.open(data_path, 'r', ctx=self.ctx) as A:
                    attr_names = A.meta['dataset'].split(',')
            for attr_name in attr_names:
                cube = self._load_data(data_path, group_dims, grid_mapping,
                                       attr_name=attr_name, handle_nan=handle_nan)
                cubes.append(cube)
        return cubes

    def _get_grid_mapping(self, data_array_path):
        """
        Get the grid mapping (Iris coord_system) from the data array metadata.
        Grid mapping is stored as a JSON string in the array meta,
        which is translated by `.grid_mappings.GridMapping`.

        """
        grid_mapping = None
        with tiledb.open(data_array_path, 'r', ctx=self.ctx) as A:
            try:
                grid_mapping_str = A.meta['grid_mapping']
            except KeyError:
                grid_mapping_str = None
        if grid_mapping_str is not None and grid_mapping_str != 'none':
            # Cannot write NoneType into TileDB array meta, so `'none'` is a
            # stand-in that must be caught.
            translator = GridMapping(grid_mapping_str)
            try:
                grid_mapping = translator.get_grid_mapping()
            except Exception as e:
                exception_type = e.__class__.__name__
                warnings.warn(f'Re-raised as warning: {exception_type}: {e}.\nGrid mapping will be None.')
        return grid_mapping

    def _get_arrays_and_dims(self, group_array_paths):
        """
        Sort the contents of a TileDB group into data arrays and dimension arrays
        by checking the metadata of each TileDB array.

        By convention of `.writers.TDBWriter.populate_array`, a data array will
        have a metadata attribute of `'dimensions'` and a dim array will have a
        metadata attribute of `'coord'`.

        """
        dim_array_paths = []
        data_array_paths = []
        for array_path in group_array_paths:
            with tiledb.open(array_path, 'r', ctx=self.ctx) as A:
                metadata = {k: v for k, v in A.meta.items()}
            if metadata.get('dataset') is not None:
                data_array_paths.append(array_path)
            elif metadata.get('coord') is not None:
                dim_array_paths.append(array_path)
            else:
                # Can't handle this!
                raise ValueError(f'Type of array at {array_path!r} not known.')
        return dim_array_paths, data_array_paths

    def _names_to_arrays(self, names):
        """Convert one or more named datasets into groups to convert to cubes."""
        if isinstance(names, str):
            names = [names]
        iter_groups = {}
        for name in names:
            # Extract a named dataset as a single Iris cube.
            named_array_path, named_array_dims = self._extract(name)
            if named_array_path.endswith('/'):
                named_array_group_path, _ = os.path.split(named_array_path[:-1])
            else:
                named_array_group_path, _ = os.path.split(named_array_path)
            if named_array_group_path in iter_groups.keys():
                iter_groups[named_array_group_path].extend(named_array_dims + [named_array_path])
            else:
                iter_groups[named_array_group_path] = named_array_dims + [named_array_path]
        result = {}
        for k, v in iter_groups.items():
            result[k] = list(set(v))
        return result

    def to_iris(self, names=None, handle_nan=None):
        """
        Convert all arrays in a TileDB into one or more Iris cubes.

        """
        self.check_groups()

        # XXX will only return the first match if more than one cube matching `name`
        # is found.
        if names is not None:
            iter_groups = self._names_to_arrays(names)
        else:
            iter_groups = self.groups

        cubes = []
        for _, group_array_paths in iter_groups.items():
            dim_paths, data_paths = self._get_arrays_and_dims(group_array_paths)
            grid_mapping = self._get_grid_mapping(data_paths[0])
            group_coords = self._load_group_dims(dim_paths, grid_mapping)
            if self.data_array_name is not None:
                group_cubes = self._load_multiattr_arrays(data_paths,
                                                          group_coords,
                                                          grid_mapping,
                                                          attr_names=names,
                                                          handle_nan=handle_nan)
            else:
                group_cubes = self._load_group_arrays(data_paths, group_coords, grid_mapping,
                                                      handle_nan=handle_nan)
            cubes.extend(group_cubes)

        self.artifact = cubes[0] if len(cubes) == 1 else CubeList(cubes)
        return self.artifact

    def to_xarray(self, names=None):
        intermediate = self.to_iris(names=names)
        self.artifact = xr.from_iris(intermediate)
        return self.artifact
