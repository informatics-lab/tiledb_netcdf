import os

import netCDF4
import numpy as np
import tiledb


class NCtoTileDB(object):
    """
    Provides an adapter class to convert a NetCDF4 dataset into a TileDB array.
    
    Outstanding questions:
      * how do we represent unlimited dimensions? (Done?)
      * how do we represent nD coordinates? (Probably OK?)
      * how do we represent scalar coordinates? (Probably OK?)
      * how do we represent auxiliary dimension-describing coordinates in tiledb?
      * how do we represent cell methods?
      * how do we represent ancilliary variables and more esoteric metadata?

    """
    data_var = {}
    shape = []
    chunks = []
    dim_coords = {}
    scalar_coords = {}
    aux_coords = {}
    bounds = {}
    grid_mapping = {}
    cell_methods = {}
    cell_measures = {}
    unlimited_dim_coords = []
    
    def __init__(self, netcdf_filename, tiledb_filepath, tiledb_array_name=None, autorun=False):
        self.netcdf_filename = netcdf_filename
        self.tiledb_filepath = tiledb_filepath
        self._tiledb_array_name = tiledb_array_name
        self.autorun = autorun
        
        self._ncds = netCDF4.Dataset(self.netcdf_filename, mode='r')

        if self._tiledb_array_name is None:
            self.array_name = os.path.basename(os.path.splitext(self.netcdf_filename)[0])
        else:
            self.array_name = self._tiledb_array_name
        self.array_filename = os.path.join(self.tiledb_filepath, self.array_name)
        
        self._ncds_dims = self._ncds.dimensions
        self._ncds_dims_names = list(self._ncds_dims.keys())      
        self._ncds_vars = self._ncds.variables
        self._ncds_vars_names = list(self._ncds.variables.keys())
        self._ncds_attrs = {key: self._ncds.getncattr(key) for key in self._ncds.ncattrs()}
        
        if self.autorun:
            self.run()

    def _check_data_var(self, variable):
        """
        Check if the variable defines the data variable. The phenomenon-describing
        variable will uniquely cover all the dimensions of the dataset itself.
        
        Unfortunately NetCDF classifies bounds as a dimension, so we need to
        ensure we don't include the bounds dimension in the comparison.
        
        """
        no_bnds_dataset_dims = set(self._ncds_dims_names) - set(['bnds'])
        variable_dims = set(variable.dimensions)
        is_data_var = len(no_bnds_dataset_dims - variable_dims) == 0
        return is_data_var
        
    def classify_variables(self):
        """
        Classify each variable as phenomenon, dim_coord, aux_coord, bounds, grid_mapping
        or unclassified (ignored).
        
        """
        # Classify unlimited dimensions.
        self.unlimited_dim_coords = [name for name in self._ncds_dims_names
                                     if self._ncds_dims[name].isunlimited()]
        
        classified_vars = []
        for var_name in self._ncds_vars_names:
            variable = self._ncds_vars[var_name]
                
            # Check if this is the data variable.
            if self._check_data_var(variable):
                classified_vars.append(var_name)
                self.data_var[var_name] = variable
                # Set chunks.
                self.chunks = variable.chunking()
                # Set shape.
                self.shape = variable.shape
                
            # Check if it's a grid mapping variable.
            elif hasattr(variable, 'grid_mapping_name'):
                self.grid_mapping[var_name] = variable
                classified_vars.append(var_name)
                
            # Check if it's a cell measure variable.
            elif hasattr(variable, 'cell_measures'):
                self.cell_measures[var_name] = variable
                classified_vars.append(var_name)
            
            # Check if it's a dimension coordinate.
            elif var_name in self._ncds_dims_names:
                self.dim_coords[var_name] = variable
                classified_vars.append(var_name)
            
            # Check if it's a scalar coordinate.
            elif len(variable.dimensions) == 0:
                self.scalar_coords[var_name] = variable
                classified_vars.append(var_name)
                
            # Check if it's a coordinate bounds.
            elif var_name.endswith('bnds'):
                self.bounds[var_name] = variable
                classified_vars.append(var_name)
                
            # TODO: check if it's a cell method variable
#             elif hasattr(variable, 'cell_measures'):
#                 pass

        # What have we missed?
        unclassified_vars = list(set(self._ncds_vars_names) - set(classified_vars))
        
        # See what we can do about the unclassified vars.
        for u_var in unclassified_vars:
            variable = self._ncds_vars[u_var]
            
            # Check if it's an auxiliary coordinate.
            for dim_var in self.dim_coords.values():
                if variable.shape == dim_var.shape:
                    self.aux_coords[u_var] = variable
                    classified_vars.append(u_var)
                    
            # Handle multidimensional coords too.
            if len(variable.shape) > 1:
                self.aux_coords[u_var] = variable
        
        # What have we still missed?
        unclassified_vars = list(set(self._ncds_vars_names) - set(classified_vars))
        
        if len(unclassified_vars):
            # We're not trying again, so just print them.
            print(f'Unclassified vars: {unclassified_vars}')

    def _get_coord(self, name):
        """Get a named coord from dim and aux coords, or error if not found."""
        coord = self.dim_coords.get(name, None)
        if coord is None:
            coord = self.aux_coords.get(name, None)
        if coord is None:
            raise ValueError(f'Coordinate {name!r} not found.')
        return coord
            
    def _coords_dims_mapping(self):
        """
        Determine the mapping of dimension indices to coordinates. This is done by
        coord length only if all coords have unique lengths. If there are
        repeat lengths then the initial order of the coords in the NetCDF's
        dimensions list is also used.
        
        TODO: support aux coords and nD coords.
        
        """
        if len(np.unique(self.shape)) != len(self.shape):
            # Handle repeated dimension lengths, e.g. shape=(100, 100, 100)
            order_specifier = self._ncds_dims_names
            dim_mapping = {order_specifier.index(dim_name): dim_name
                           for dim_name in self.dim_coords.keys()}
        else:
            # All dimension lengths unique.
            order_specifier = self.shape
            dim_mapping = {order_specifier.index(self._get_coord(dim_name).shape[0]): dim_name
                           for dim_name in self.dim_coords.keys()}
            
        # dim_mapping ---> {index: dim_name}
        # coord_mapping -> {dim_name: index}
        coord_mapping = {v:k for k, v in dim_mapping.items()}
        return dim_mapping, coord_mapping
    
    def _create_tdb_dim(self, dim_name, chunks):
        dim_coord = self.dim_coords[dim_name]
        dtype = dim_coord.dtype
        
        # TODO: work out nD coords.
        dim_coord_len, = dim_coord.shape
        
        # Set the tdb dimension dtype to `int64` regardless of input.
        # All tdb dims in a domain must have exactly the same dtype.
        dim_dtype = np.int64
        
        # Sort out the domain, based on whether the dim is unlimited.
        if dim_name in self.unlimited_dim_coords:
            domain_max = np.iinfo(dim_dtype).max - dim_coord_len
        else:
            domain_max = dim_coord_len
        
        return tiledb.Dim(name=dim_name,
                          domain=(0, domain_max),
                          tile=chunks,
                          dtype=dim_dtype)
        
    def create_tiledb_array(self):
        dim_mapping, coord_mapping = self._coords_dims_mapping()
        
        # Create tdb array dimensions.
        ordered_dims = sorted(self.dim_coords.keys(),
                              key=lambda name: coord_mapping[name])
        array_dims = []
        for dim_name in ordered_dims:
            dim_index = coord_mapping[dim_name]
            tdb_dim = self._create_tdb_dim(dim_name, self.chunks[dim_index])
            array_dims.append(tdb_dim)
        
        # Create tdb array domain.
        domain = tiledb.Domain(*array_dims)
        
        # Create array attribute.
        (phenom_name, data_var), = self.data_var.items()
        phenom = tiledb.Attr(name=phenom_name, dtype=data_var.dtype)
        
        # Create an empty array.
        schema = tiledb.ArraySchema(domain=domain, sparse=False, attrs=[phenom])
        tiledb.Array.create(self.array_filename, schema)
        
    def _array_indices(self, data_var_shape, start_index=0):
        """Set the array indices to write the array data into."""
        array_indices = []
        for dim in data_var_shape:
            array_indices.append(slice(start_index, dim))
        return tuple(array_indices)
        
    def populate_array(self, start_index=0): 
        with tiledb.open(self.array_filename, 'w') as A:
            # Add data array.
            data_var, = self.data_var.values()
            write_indices = self._array_indices(data_var.shape, start_index)
            A[write_indices] = data_var[...]
            
            # Add metadata.
            for k, v in self._ncds_attrs.items():
                A.meta[k] = v
            
    def run(self):
        self.classify_variables()
        self.create_tiledb_array()
        self.populate_array()
        