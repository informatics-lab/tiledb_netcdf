import os

import netCDF4
import numpy as np
import tiledb


class NCDataModel(object):
    """
    Provides an adapter class to convert a NetCDF4 dataset into Python objects.
    
    Outstanding questions:
      * how do we represent unlimited dimensions? (Done)
      * how do we represent nD coordinates? (Probably OK?)
      * how do we represent scalar coordinates? (Probably OK?)
      * how do we represent auxiliary dimension-describing coordinates in tiledb? (OK?)
      * how do we represent cell methods? (not done)
      * how do we represent ancilliary variables and more esoteric metadata? (not done)

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
    
    def __init__(self, netcdf_filename):
        self.netcdf_filename = netcdf_filename
        
        self._ncds = netCDF4.Dataset(self.netcdf_filename, mode='r')
        
        self._ncds_dims = self._ncds.dimensions
        self._ncds_dims_names = list(self._ncds_dims.keys())      
        self._ncds_vars = self._ncds.variables
        self._ncds_vars_names = list(self._ncds.variables.keys())
        self._ncds_attrs = {key: self._ncds.getncattr(key) for key in self._ncds.ncattrs()}

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
                # Set shape.
                self.shape = variable.shape
                # Set chunks.
                chunks = variable.chunking()
                if chunks == 'contiguous':
                    chunks = self.shape
                self.chunks = chunks
                
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
        