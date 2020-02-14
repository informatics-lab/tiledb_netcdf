# tiledb_netcdf
An adapter to convert NetCDF files to [TileDB](https://tiledb.com/) or [Zarr](https://zarr.readthedocs.io/en/stable/index.html) arrays.

## Usage Example

Demonstrate using this library to convert NetCDF files, and read the results.

### 1. Create a common data model

Use the data model class `NCDataModel` to create a data model:

```python
from nctotdb import NCDataModel

my_nc_filepath = '/path/to/my/file.nc'
data_model = NCDataModel(my_nc_file)

# Classify individual variables in the NetCDF file as data variables, dimension variables, etc...
data_model.classify_variables()
# Also set extra metadata about the NetCDF file in general, such as unique enclosing domains.
data_model.get_metadata()
```

### 2. Write the data model

With a data model created, we can convert the contents of the NetCDF file to a supported
cloud-ready fileformat; namely one of `TileDB` or `Zarr`:

```python
from nctotdb import TDBWriter, ZarrWriter

# TileDB.
tdb_filepath = '/path/to/my_tdb'
tdb_writer = TDBWriter(data_model, tdb_filepath)
# Or, if you know you're going to need to append to the `z` dimension... 
long_z_tdb_writer = TDBWriter(data_model, tdb_filepath, unlimited_dims='z')
tdb_writer.create_domains()

# Zarr.
zarr_filepath = '/path/to/my_zarr'
zarr_writer = ZarrWriter(data_model, zarr_filepath)
zarr_writer.create_zarr()
```

#### 2a. Append

We can also append the contents of a named array along a named dimension:
```python
# Create a data model for another NetCDF file.
my_other_nc_filepath = '/path/to/my/other_file.nc'
other_data_model = NCDataModel(my_other_nc_file)

# Append (TileDB).
tdb_writer.append(other_data_model, 'array_name', 'dimension_name')

# Append (Zarr).
zarr_writer.append(other_data_model, 'array_name', 'dimension_name')
```

### 3. Read Converted Arrays

We can use the `Reader` classes to read our TileDB or Zarr arrays using Iris or Xarray:

```python
from nctotdb import TDBReader, ZarrReader

tdb_reader = TDBReader('/path/to/my_tdb')
zarr_reader = ZarrReader('/path/to/my_zarr')

# TileDB to Iris.
cubes = tdb_reader.to_iris()  # Convert all TileDB arrays to Iris Cubes.
cube = tdb_reader.to_iris('array_name')  # Convert a named array to an Iris Cube.

# TileDB to Xarray.
dss = tdb_reader.to_xarray()  # Convert all TileDB arrays to Xarray.
ds = tdb_reader.to_xarray('array_name')  # Convert a named array to an Xarray dataset.

# Zarr to Iris or Xarray.
cubes = zarr_reader.to_iris()
ds = zarr_reader.to_xarray()
```