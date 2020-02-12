from itertools import chain
import os

import cf_units
import dask.array as da
from iris.coords import CellMethod, DimCoord
from iris.cube import Cube, CubeList
from iris.fileformats.netcdf import parse_cell_methods
import tiledb
import xarray as xr
import zarr


class Reader(object):
    """
    Abstract reader class that defines the API.

    TODO replace all os usages with tiledb ls'.

    """
    def __init__(self, input_filepath):
        self.input_filepath = input_filepath

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


class TDBReader(Reader):
    def __init__(self, input_filepath, storage_options=None):
        super().__init__(input_filepath)

        self.storage_options = storage_options
        self.groups = {}
        self._arrays = None

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

    def _array_paths(self):
        """Produce a mapping of array name to array path irrespective of groups."""
        self.check_groups()
        all_paths = chain.from_iterable([paths for paths in self.groups.values()])
        arrays = {}
        for path in all_paths:
            _, array_name = os.path.split(path)
            # XXX assumes that we will not have duplicated array names.
            arrays[array_name] = path
        self.arrays = arrays

    def _tdb_walker(self):
        group = None
        def inner(itm_path, itm_type):
            nonlocal group
            if itm_type == 'group':
                group = itm_path
                self.groups[group] = []
            else:
                self.groups[group].append(itm_path)
        return inner

    def get_groups_and_arrays():
        classifier = self._tdb_walker()
        tiledb.walk(self.input_filepath,
                    lambda itm_path, itm_type: classifier(obj_path, itm_type))

    def tdb_dir_contents(self, dir):
        contents = []
        tiledb.ls(self.input_filepath, lambda obj_path, _: contents.append(obj_path))
        return contents

    def _get_dim_coords(self, array_filepath):
        with tiledb.open(array_filepath, 'r') as A:
            dims_string = A.meta['dimensions']
        return dims_string.split(',')

    def _map_coords_inds(self, group_name, dim_name, spatial_inds):
        """
        Map coordinate point values for a named dimension to indices in that
        dimension and return as a slice of indices.

        TODO handle time coordinates (by loading the calendar from the array meta?).

        """
        dim_filepath = os.path.join(self.input_filepath, group_name, dim_name)
        with tiledb.open(dim_filepath, 'r') as D:
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
        array_filepath = os.path.join(self.input_filepath, group_name, array_name)
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

        with tiledb.open(array_filepath, 'r') as A:
            subarray = A[tuple(indices)]
        return subarray

    def _extract(self, array_name):
        """
        Return the path to a named array, plus paths for all the associated
        dimension arrays.

        """
        # Sanity check the requested array name is in this TileDB.
        assert array_name in self.arrays.keys()

        named_array_path = self.arrays[array_name]
        named_group_path, _ = os.path.split(named_array_path)
        named_group_arrays = self.groups[named_group_path]

        with tiledb.open(named_array_path, 'r') as A:
            dim_names = A.meta['dimensions'].split(',')

        dim_paths = []
        for dim_name in dim_names:
            for array_path in named_group_arrays:
                if array_path.endswith(dim_name):
                    dim_paths.append(array_path)
                    break
        # Confirm we have an array path for each dim_name.
        assert len(dim_paths) == len(dim_names)

        return named_array_path, dim_paths

    def _load_dim(self, dim_path):
        """
        Create an Iris DimCoord from a TileDB array describing a dimension.
        
        # TODO not handled here: circular, coord_system.

        """
        with tiledb.open(dim_path, 'r') as D:
            metadata = D.meta
            points = D[:]

        coord_name = metadata.pop('coord')
        standard_name = metadata.pop('standard_name')
        long_name = metadata.pop('long_name')
        var_name = metadata.pop('long_name')

        units = metadata.pop('units')
        if coord_name == 'time':
            # Handle special-case complicated time coords.
            calendar = metadata.pop('calendar')
            units = cf_units.Unit(units, calendar=calendar)

        coord = DimCoord(points, 
                         standard_name=coord_name,
                         long_name=long_name,
                         units=units,
                         attributes={k: v for k, v in metadata.items()})
        return coord_name, coord

    def _load_group_dims(self, group_dim_paths):
        """Load all dimension-describing (coordinate) arrays in the group."""
        group_dims = {}
        for dim_path in group_dim_paths:
            name, coord = self._load_dim(dim_path)
            group_dims[name] = coord
        return group_dims

    def _load_data(self, array_path, group_dims):
        """
        Create an Iris cube from a TileDB array describing a data variable and
        pre-loaded dimension-describing coordinates.

        TODO not handled here: aux coords and dims, cell measures, aux factories.

        """
        with tiledb.open(array_path, 'r') as A:
            metadata = A.meta
            attr_name = metadata.pop('dataset')
            lazy_data = da.from_tiledb(array_path,
                                       attribute=attr_name,
                                       storage_options=self.storage_options)

        cell_methods = parse_cell_methods(metadata.pop('cell_methods'))
        dims = metadata.pop('dimensions').split(',')
        # Dim Coords And Dims (mapping of coords to cube axes).
        dcad = [(group_dims[name], i) for i, name in enumerate(dims)]

        return Cube(lazy_data,
                    standard_name=metadata.pop('standard_name'),
                    long_name=metadata.pop('long_name'),
                    var_name=metadata.pop('var_name'),
                    units=metadata.pop('units'),
                    dim_coords_and_dims=dcad,
                    cell_methods=cell_methods,
                    attributes={k: v for k, v in metadata.items()})

    def _load_group_arrays(self, data_paths, group_dims):
        """Load all data-describing (cube) arrays in the group."""
        cubes = []
        for data_path in data_paths:
            cube = self._load_data(data_path, group_dims)
            cubes.append(cube)
        return cubes

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
            with tiledb.open(array_path, 'r') as A:
                data_array = A.meta.get('dataset')
                dim_array = A.meta.get('coord')
            if data_array is not None:
                data_array_paths.append(array_path)
            elif dim_array is not None:
                dim_array_paths.append(array_path)
            else:
                # Can't handle this!
                raise ValueError(f'Type of array at {array_path!r} not known.')

        return dim_array_paths, data_array_paths

    def to_iris(self, name=None):
        """
        Convert all arrays in a TileDB into one or more Iris cubes.

        """
        # Loop through groups in the TileDB:
        # In each group, make an Iris coordinate of each of the coord arrays in the group,
        # pulling metadata out of the coord array's meta,
        # and a coords and dims mapping of [axis, coordinate name].
        # Make a cube for each data array, wrapping the TileDB array values in dask,
        # adding the appropriate coords and dims mapping, and pulling metadata out of the
        # data array's meta (note that cell methods and STASH will be special cases).
        # Add all discrete cubes to a cubelist and return.
        self.check_groups()

        if name is not None:
            named_array_path, named_array_dims = self._extract(name)
            named_array_group_path, _ = os.path.split(named_array_path)
            iter_groups = {named_array_group_path: named_array_dims}
        else:
            iter_groups = self.groups

        cubes = []
        for group_path, group_array_paths in iter_groups.items():
            dim_paths, data_paths = self._get_arrays_and_dims(group_array_paths)
            group_coords = self._load_group_dims(dim_paths)
            group_cubes = self._load_group_arrays(data_paths)
            cubes.extend(group_cubes)

        self.artifact = cubes[0] if len(cubes) == 1 else CubeList(cubes)
        return self.artifact

    def to_xarray(self):
        intermediate = self.to_iris()
        self.artifact = xr.from_iris(intermediate)
        return self.artifact


class ZarrReader(Reader):
    def __init__(self, input_filepath):
        super().__init__(input_filepath)

    def to_iris(self):
        intermediate = self.to_xarray()
        self.artifact = intermediate.to_iris()
        return self.artifact

    def to_xarray(self):
        self.artifact = xr.open_zarr(self.input_filepath)
        return self.artifact