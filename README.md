# tiledb_netcdf
An adapter to convert [NetCDF](https://www.unidata.ucar.edu/software/netcdf/) files to [TileDB](https://tiledb.com/) or [Zarr](https://zarr.readthedocs.io/en/stable/index.html) arrays.

## Usage Examples

Here we'll demonstrate using this library to convert NetCDF files to TileDB and Zarr, and read the results using Iris and Xarray.

### Converting to TileDB

TileDB supports direct interaction with blob storage as provided by public cloud platforms.
Currently this library only supports Azure Blob Storage containers, but we will add AWS support
in a future release. If you'd like that support right now, do raise an issue 👆!

#### 1. Create a common data model

Use the data model class `NCDataModel` to create a data model:

```python
from nctotdb import NCDataModel

data_model = NCDataModel('/path/to/my/file.nc')
```

#### 2. Write the data model

With a data model created, we can convert the contents of the NetCDF file to a supported
cloud-ready fileformat (one of `TileDB` or `Zarr`). We'll demonstrate writing to TileDB here,
with Zarr covered in the next section.

As mentioned earlier, we can write either to posix-like filepaths or directly to
an Azure Storage account blob container. Let's cover each of these in turn.

##### 2a. Filepaths

```python
from nctotdb import TileDBWriter

# TileDB.
tiledb_save_path = '/path/to/my_tdb'
tiledb_name = 'my_tiledb'
unlimited_dims = 'z'  # Useful if you know you're going to need to append to the `z` dimension

with data_model.classify():
    writer = TileDBWriter(data_model,
                          array_filepath=tiledb_save_path,
                          array_name=tiledb_name,
                          unlimited_dims=unlimited_dims)
    writer.create_domains()
```

##### 2b. Blob container

Some more setup is needed to interface with an Azure Storage account Blob container.
We need to provide authentication to the Azure Storage account and configure TileDB operations
to work with the Blob container we wish to write to:

```python
import tiledb

# Azure blob storage definitions.
storage_account_name = 'my_azure_storage_account_name'
container = 'my_blob_container_name'
uri = f'azure://{container}'
access_key = 'my_azure_access_key'

# TileDB configuration for Azure Blob.
cfg = tiledb.Config()
cfg['vfs.azure.storage_account_name'] = storage_account_name
cfg['vfs.azure.storage_account_key'] = access_key
cfg["vfs.s3.use_multipart_upload"] = 'false'

ctx = tiledb.Ctx(config=cfg)
```

**Important!** Do not share or publish your Azure Storage account key! You can also
set an environment variable that TileDB will use instead of pasting your account key
into your code.

Now we can write to our TileDB array. This is much the same as with posix-like
paths, other than that we must also pass the TileDB `Ctx` (context) object and specify
a container rather than a filepath to save to:

```python
with data_model.classify():
    writer = TileDBWriter(data_model,
                          container=container,
                          array_name=tiledb_name,
                          unlimited_dims=unlimited_dims,
                          ctx=ctx)
    writer.create_domains()
```

#### 3. Append

We can also append the contents of one or more extra NetCDF files along a named dimension.
The extra NetCDF files can be specified either as a list of filepaths or as a list of data model
objects. If filepaths are specified they will be automatically converted to data model objects.

```python
append_files = ['file1.nc', 'file2.nc', 'file3.nc']
data_array_name = 'data'  # The name of the data arrays in the TileDB array, typically `data`.

with data_model.classify():
    writer.append(append_files, unlimited_dims, data_array_name)
```

#### 4. Read Converted Arrays

We can use the `Reader` classes to read our TileDB or Zarr arrays using Iris or Xarray:

```python
from nctotdb import TDBReader, ZarrReader

tiledb_reader = TileDBReader('/path/to/my_tdb')

# TileDB to Iris.
cubes = tiledb_reader.to_iris()  # Convert all TileDB arrays to Iris Cubes.
cube = tiledb_reader.to_iris('array_name')  # Convert a named variable to an Iris Cube.

# TileDB to Xarray.
dss = tiledb_reader.to_xarray()  # Convert all TileDB arrays to Xarray.
ds = tiledb_reader.to_xarray('array_name')  # Convert a named variable to an Xarray dataset.
```

### Converting to Zarr

We can also convert NetCDF files to Zarr using this library, and read these Zarrs
back into Iris and Xarray. A similar set of APIs is provided for Zarr as was provided
for TileDB.

#### 1. Create a common data model

This is exactly the same as for TileDB. The differentiation comes at the next step
when we choose the data representation format we want to use to store the contents of
the NetCDF file represented by the data model.

```python
from nctotdb import NCDataModel

my_nc_filepath = '/path/to/my/file.nc'
data_model = NCDataModel(my_nc_file)
```

#### 2. Write to Zarr

With a data model created we can write the contents of the NetCDF file as exposed via
the data model. Here we write the contents to Zarr:

```python
from nctotdb import ZarrWriter

with data_model.classify():
    zarr_writer = ZarrWriter(data_model, '/path/to/my_zarr',
                             array_name='my_zarr')
    zarr_writer.create_zarr()
```

#### 3. Append

We can also add the contents of other NetCDF files to the Zarr we created, and
extend one of the Zarr's dimensions:

```python
my_other_nc_filepath = '/path/to/my/other_file.nc'
other_data_model = NCDataModel(my_other_nc_file)

append_var_name = 'array_name'
append_dim = 'dimension_name'

with other_data_model.classify():
    zarr_writer.append(other_data_model, append_var_name, append_dim)
```

#### 4. Read Zarr

And finally we can read the Zarr we created into Iris and Xarray:

```python
from nctotdb import ZarrReader

zarr_reader = ZarrReader('/path/to/my_zarr')
cubes = zarr_reader.to_iris()
ds = zarr_reader.to_xarray()
```
