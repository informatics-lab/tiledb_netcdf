import os

import numpy as np
import tiledb
import zarr


class TDBWriter(object):
    """
    Provides a class to write Python objects loaded from NetCDF to TileDB.
    
    Data Model: an instance of `NCDataModel` supplying data from a NetCDF file.
    Filepath: the filepath to save the tiledb array at.
    
    """
    def __init__(self, data_model, tiledb_filepath, tiledb_array_name=None):
        self.data_model = data_model
        
        self.tiledb_filepath = tiledb_filepath
        self._tiledb_array_name = tiledb_array_name
        
        if self._tiledb_array_name is None:
            self.array_name = os.path.basename(os.path.splitext(self.data_model.netcdf_filename)[0])
        else:
            self.array_name = self._tiledb_array_name
        self.array_filename = os.path.join(self.tiledb_filepath, self.array_name)

    def _get_coord(self, name):
        """Get a named coord from dim and aux coords, or error if not found."""
        coord = self.data_model.dim_coords.get(name, None)
        if coord is None:
            coord = self.data_model.aux_coords.get(name, None)
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
        if len(np.unique(self.data_model.shape)) != len(self.data_model.shape):
            # Handle repeated dimension lengths, e.g. shape=(100, 100, 100)
            order_specifier = self.data_model._ncds_dims_names
            dim_mapping = {order_specifier.index(dim_name): dim_name
                           for dim_name in self.data_model.dim_coords.keys()}
        else:
            # All dimension lengths unique.
            order_specifier = self.data_model.shape
            dim_mapping = {order_specifier.index(self._get_coord(dim_name).shape[0]): dim_name
                           for dim_name in self.data_model.dim_coords.keys()}
            
        # dim_mapping ---> {index: dim_name}
        # coord_mapping -> {dim_name: index}
        coord_mapping = {v:k for k, v in dim_mapping.items()}
        return dim_mapping, coord_mapping
    
    def _create_tdb_dim(self, dim_name, chunks):
        dim_coord = self.data_model.dim_coords[dim_name]
        dtype = dim_coord.dtype
        
        # TODO: work out nD coords.
        dim_coord_len, = dim_coord.shape
        
        # Set the tdb dimension dtype to `int64` regardless of input.
        # All tdb dims in a domain must have exactly the same dtype.
        dim_dtype = np.int64
        
        # Sort out the domain, based on whether the dim is unlimited.
        if dim_name in self.data_model.unlimited_dim_coords:
            domain_max = np.iinfo(dim_dtype).max - dim_coord_len
        else:
            domain_max = dim_coord_len

        return tiledb.Dim(name=dim_name,
                          domain=(0, domain_max),
                          tile=chunks,
                          dtype=dim_dtype)

    def _create_tdb_attrs(self):
        # Create array attribute.
        tdb_attrs = []
        for phenom_name in self.data_model.data_var.keys():
            data_var = self.data_model.data_var[phenom_name]
            phenom = tiledb.Attr(name=phenom_name, dtype=data_var.dtype)
            tdb_attrs.append(phenom)
        return tdb_attrs
    
    def create_array(self):
        dim_mapping, coord_mapping = self._coords_dims_mapping()
        
        # Create tdb array dimensions.
        ordered_dims = sorted(self.data_model.dim_coords.keys(),
                              key=lambda name: coord_mapping[name])

        array_dims = []
        for dim_name in ordered_dims:
            dim_index = coord_mapping[dim_name]
            tdb_dim = self._create_tdb_dim(dim_name, self.data_model.chunks[dim_index])
            array_dims.append(tdb_dim)
        
        # Create tdb array domain.
        domain = tiledb.Domain(*array_dims)
        
        # Get tdb attributes.
        attrs = self._create_tdb_attrs()
   
        # Create an empty array.
        schema = tiledb.ArraySchema(domain=domain, sparse=False, attrs=attrs)
        tiledb.Array.create(self.array_filename, schema)
        
    def _array_indices(self, start_index=0):
        """Set the array indices to write the array data into."""
        array_indices = []
        for dim in self.data_model.shape:
            array_indices.append(slice(start_index, dim))
        return tuple(array_indices)
        
    def populate_array(self, start_index=0): 
        with tiledb.open(self.array_filename, 'w') as A:
            # Add data array.
            write_indices = self._array_indices(start_index)
            attribute_mapping = {name: var[...] for name, var in self.data_model.data_var.items()}
            A[write_indices] = attribute_mapping

            # Add metadata.
            for k, v in self.data_model._ncds_attrs.items():
                A.meta[k] = v

                
class ZarrWriter(object):
    """
    Provides a class to write Python objects loaded from NetCDF to zarr.
    
    TODO:
      * Support groups
      * Labelled dimensions / support for coords.
    
    """
    def __init__(self, data_model, filepath, array_name=None):
        self.data_model = data_model
        self.filepath = filepath
        self._array_name = array_name
        
        if self._array_name is None:
            self.array_name = os.path.basename(os.path.splitext(self.data_model.netcdf_filename)[0])
        else:
            self.array_name = self._array_name
        self.array_filename = f'{os.path.join(os.path.abspath("."), self.filepath, self.array_name)}.zarr'
        print(self.array_filename)
        
        self.zarray = None
    
    def create_array(self):
        self.zarray = zarr.open(self.array_filename,
                                shape=self.data_model.shape,
                                mode='a',
                                chunks=self.data_model.chunks)
    
    def populate_array(self):
        self.zarray[:] = list(self.data_model.data_var.values())[0][...]
        
        # Add metadata.
        for k, v in self.data_model._ncds_attrs.items():
            self.zarray.attrs[k] = v