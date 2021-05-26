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
data_model.populate()
```

##### Manually modifying classification

Classifying NetCDF variables is a long way from a precise science, and occasionally the
process may fail to correctly classify a variable. In such a case you can manually modify
the classification processes by using the following instead of calling `data_model.populate()`:

```python
data_model = NCDataModel('/path/to/my/file.nc')
my_bespoke_data_var_name = 'foobarbaz'

with data_model.open_netcdf():
    data_model.classify_variables()
    data_model.data_var_names = [my_bespoke_data_var_name]
    data_model.get_metadata()
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

writer.append(append_files, unlimited_dims, data_array_name)
```

You can track the progress of append operations by enabling verbose mode:

```python
writer.append(append_files, unlimited_dims, data_array_name,
              verbose=True)
```

If you have a large number of files to append or you simply want the append to complete
faster, `tiledb_netcdf` can use dask to parallelise the append operation on a per-file basis.
Assuming you have already set up a dask cluster, `my_cluster`:

```python
client = dask.distributed.Client(my_cluster)
logfile = "append.log"
writer.append(append_files, unlimited_dims, data_array_name,
              parallel=True, logfile=logfile)
```

**Note:** it is recommended you also log parallel appends for error tracking, should
anything go wrong during the append process.

##### 3a. Scalar Append

One case of appending needs to be handled differently. This is the case where the datasets to
be appended are scalar along the append dimension. For example, you may wish to append along the
`time` dimension, but the base dataset and all files to be appended only contain a single
(that is, scalar) time point. In this case a scalar append needs to be carried out.

Typically the append algorithm uses the separation between points along the append dimension
to calculate the offsets of all datasets to be appended. With only a single point along the
append dimension this is not possible, so instead you need to also supply a file to the append
call that allows the offset between files to be calculated. To ensure the correct offset is
calculated, this file *must* describe the next step in the append dimension from the file
originally used to create the TileDB array.

The file used to calculate the offset is passed into the append operation using the
`baseline` keyword argument. For example:

```python
append_files = ['file1.nc', 'file2.nc', 'file3.nc', 'file4.nc', 'file5.nc']
data_array_name = 'data'

writer.append(append_files, unlimited_dims, data_array_name,
              baselines={unlimited_dims: append_files[0]})
```

**Note:** The file used to calculate the offsets is not appended as well as being used to calculate
the offset. You will need to include the offset file in the append files as well!

**Note:** All appends with a scalar append dimension must be supplied with a `baseline`
file to calculate the offset, even if an append has already successfully been carried out. You must also specify one baseline file per scalar append dimension, in a dictionary of `{"append_dim": baseline_file}`.

If you try and perform an append along a scalar dimension without providing a `baseline`
file to calculate the offset, you will encounter an error message:

```python-traceback
ValueError: Cannot determine scalar step without a baseline dataset.
```

##### 3b. Custom offsets between files being appended

You may occasionally need to override the offset between successive files being appended, for example to introduce some padding between files, or to handle unexpected short files. This can be done using
the `override_offsets` kwarg to `append`. As with specifying `baselines`, you need to pass a dictionary linking the named append dimension to the offset override you wish to apply to that dimension. For example:

```python
append_files = ['file1.nc', 'file2.nc', 'file3_short.nc', 'file4.nc', 'file5.nc']
expected_dim_len = 10
data_array_name = 'data'

writer.append(append_files, unlimited_dims, data_array_name,
              override_offsets={unlimited_dims: expected_dim_len})
```

In this case, the third file is shorter than expected (as helpfully indicated in its filename), and the override offset allows us to pad the append dimension with missing data where the file runs short. We can use the `fill_missing_points` method to fill in the gap in the associated dimension coordinate after the append has completed:

```python
writer.fill_missing_points(unlimited_dims)
```

**Note:** you do not have to provide an override offset for every append dimension.

#### 4. Read Converted Arrays

We can use the `Reader` classes to read our TileDB array with Iris or Xarray:

```python
from nctotdb import TileDBReader

# From a posix-like filepath:
tiledb_reader = TileDBReader(tiledb_name, array_filepath=tiledb_save_path)

# Or directly from Azure Blob:
tiledb_reader = TileDBReader(tiledb_name, container=container, ctx=ctx)

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
data_model.populate()
```

#### 2. Write to Zarr

With a data model created we can write the contents of the NetCDF file as exposed via
the data model. Here we write the contents to Zarr:

```python
from nctotdb import ZarrWriter

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

zarr_writer.append(other_data_model, append_var_name, append_dim)
```

#### 4. Read Zarr

Finally we can read the Zarr we created into Iris and Xarray:

```python
from nctotdb import ZarrReader

zarr_reader = ZarrReader('/path/to/my_zarr')
cubes = zarr_reader.to_iris()
ds = zarr_reader.to_xarray()
```
