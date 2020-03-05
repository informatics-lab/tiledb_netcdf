import json

import iris.coord_systems


def store_grid_mapping(grid_mapping_variable):
    """Store the metadata for a NetCDF grid mapping variable as a JSON string."""
    ncattrs = grid_mapping_variable.ncattrs()
    return json.dumps({k: grid_mapping_variable.getncattr(k) for k in ncattrs})


# Taken from https://github.com/SciTools/iris/blob/master/lib/iris/fileformats/_pyke_rules/fc_rules_cf.krb.
class GridMapping(object):
    """
    Convert a NetCDF grid mapping variable (expressed as a JSON string) into an
    Iris coordinate system.

    """
    # Grid mapping names.
    GRID_MAPPING_ALBERS = 'albers_conical_equal_area'
    GRID_MAPPING_AZIMUTHAL = 'azimuthal_equidistant'
    GRID_MAPPING_LAMBERT_AZIMUTHAL = 'lambert_azimuthal_equal_area'
    GRID_MAPPING_LAMBERT_CONFORMAL = 'lambert_conformal_conic'
    GRID_MAPPING_LAMBERT_CYLINDRICAL = 'lambert_cylindrical_equal_area'
    GRID_MAPPING_LAT_LON = 'latitude_longitude'
    GRID_MAPPING_MERCATOR = 'mercator'
    GRID_MAPPING_ORTHO = 'orthographic'
    GRID_MAPPING_POLAR = 'polar_stereographic'
    GRID_MAPPING_ROTATED_LAT_LON = 'rotated_latitude_longitude'
    GRID_MAPPING_STEREO = 'stereographic'
    GRID_MAPPING_TRANSVERSE = 'transverse_mercator'
    GRID_MAPPING_VERTICAL = 'vertical_perspective'

    # Earth grid attributes.
    SEMI_MAJOR_AXIS = 'semi_major_axis'
    SEMI_MINOR_AXIS = 'semi_minor_axis'
    INVERSE_FLATTENING = 'inverse_flattening'
    EARTH_RADIUS = 'earth_radius'
    LON_OF_PROJ_ORIGIN = 'longitude_of_projection_origin'
    NORTH_POLE_LAT = 'grid_north_pole_latitude'
    NORTH_POLE_LON = 'grid_north_pole_longitude'
    NORTH_POLE_GRID_LON = 'north_pole_grid_longitude'

    def __init__(self, grid_mapping_string):
        self.grid_mapping_string = grid_mapping_string
        self.grid_mapping = json.loads(self.grid_mapping_string)

    def _get_ellipsoid(self):
        """Return the ellipsoid definition."""
        major = self.grid_mapping.get(self.SEMI_MAJOR_AXIS, None)
        minor = self.grid_mapping.get(self.SEMI_MINOR_AXIS, None)
        inverse_flattening = self.grid_mapping.get(self.INVERSE_FLATTENING, None)

        if major is not None and minor is not None:
            inverse_flattening = None
        if major is None and minor is None and inverse_flattening is None:
            major = self.grid_mapping.get(self.EARTH_RADIUS, None)

        return major, minor, inverse_flattening

    def build_coordinate_system(self):
        """Create a coordinate system from the grid mapping variable."""
        major, minor, inverse_flattening = self._get_ellipsoid()
        return iris.coord_systems.GeogCS(major, minor, inverse_flattening)

    def build_rotated_coordinate_system(self):
        """Create a rotated coordinate system from the grid mapping variable."""
        major, minor, inverse_flattening = self._get_ellipsoid()
        
        north_pole_latitude = self.grid_mapping.get(self.NORTH_POLE_LAT, 90.0)
        north_pole_longitude = self.grid_mapping.get(self.NORTH_POLE_LON, 0.0)
        north_pole_grid_lon = self.grid_mapping.get(self.NORTH_POLE_GRID_LON, 0.0)
        
        ellipsoid = None
        if major is not None or minor is not None or inverse_flattening is not None:
            ellipsoid = iris.coord_systems.GeogCS(major, minor, inverse_flattening)

        return iris.coord_systems.RotatedGeogCS(north_pole_latitude, north_pole_longitude,
                                                north_pole_grid_lon, ellipsoid)

    def build_mercator_coordinate_system(self):
        """Create a Mercator coordinate system from the grid mapping variable."""
        major, minor, inverse_flattening = self._get_ellipsoid()
        longitude_of_projection_origin = self.grid_mapping.get(self.LON_OF_PROJ_ORIGIN, None)

        ellipsoid = None
        if major is not None or minor is not None or inverse_flattening is not None:
            ellipsoid = iris.coord_systems.GeogCS(major, minor, inverse_flattening)

        return iris.coord_systems.Mercator(longitude_of_projection_origin, ellipsoid=ellipsoid)

    def get_grid_mapping(self):
        """
        Determine the type of grid mapping variable we have and call the
        appropriate converter.

        Note that we do not support translation of all grid mapping variables.
        Only the following are supported:
          * latitude_longitude
          * mercator
          * rotated_latitude_longitude
        
        """
        grid_mapping_name = self.grid_mapping.get('grid_mapping_name').lower()

        if grid_mapping_name == self.GRID_MAPPING_LAT_LON:
            result = self.build_coordinate_system()
        elif grid_mapping_name == self.GRID_MAPPING_ROTATED_LAT_LON:
            result = self.build_rotated_coordinate_system()
        elif grid_mapping_name == self.GRID_MAPPING_MERCATOR:
            result = self.build_mercator_coordinate_system()
        elif grid_mapping_name == self.GRID_MAPPING_ALBERS:
            # Grid mapping type not handled yet.
            raise NotImplementedError(f'Grid mapping name {grid_mapping_name} is not currently supported.')
        elif grid_mapping_name == self.GRID_MAPPING_AZIMUTHAL:
            # Grid mapping type not handled yet.
            raise NotImplementedError(f'Grid mapping name {grid_mapping_name} is not currently supported.')
        elif grid_mapping_name == self.GRID_MAPPING_LAMBERT_AZIMUTHAL:
            # Grid mapping type not handled yet.
            raise NotImplementedError(f'Grid mapping name {grid_mapping_name} is not currently supported.')
        elif grid_mapping_name == self.GRID_MAPPING_LAMBERT_CONFORMAL:
            # Grid mapping type not handled yet.
            raise NotImplementedError(f'Grid mapping name {grid_mapping_name} is not currently supported.')
        elif grid_mapping_name == self.GRID_MAPPING_LAMBERT_CYLINDRICAL:
            # Grid mapping type not handled yet.
            raise NotImplementedError(f'Grid mapping name {grid_mapping_name} is not currently supported.')
        elif grid_mapping_name == self.GRID_MAPPING_ORTHO:
            # Grid mapping type not handled yet.
            raise NotImplementedError(f'Grid mapping name {grid_mapping_name} is not currently supported.')
        elif grid_mapping_name == self.GRID_MAPPING_POLAR:
            # Grid mapping type not handled yet.
            raise NotImplementedError(f'Grid mapping name {grid_mapping_name} is not currently supported.')
        elif grid_mapping_name == self.GRID_MAPPING_STEREO:
            # Grid mapping type not handled yet.
            raise NotImplementedError(f'Grid mapping name {grid_mapping_name} is not currently supported.')
        elif grid_mapping_name == self.GRID_MAPPING_TRANSVERSE:
            # Grid mapping type not handled yet.
            raise NotImplementedError(f'Grid mapping name {grid_mapping_name} is not currently supported.')
        elif grid_mapping_name == self.GRID_MAPPING_VERTICAL:
            # Grid mapping type not handled yet.
            raise NotImplementedError(f'Grid mapping name {grid_mapping_name} is not currently supported.')
        else:
            # Not a valid grid mapping name.
            raise ValueError(f'{grid_mapping_name!r} is not a valid grid mapping name.')

        return result
