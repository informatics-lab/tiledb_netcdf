from collections import namedtuple
import json
import logging
import os

import tiledb

from ..data_model import NCDataModel
from .. import utils


append_arg_list = ['other', 'domain', 'name', 'axis', 'dim',
                   'ind_stop', 'dim_stop', 'step', 'scalar',
                   'mapping', 'verbose', 'job_number', 'n_jobs',
                   'make_data_model', 'offset', 'ctx', 'logfile']
defaults = [None] * len(append_arg_list)
AppendArgs = namedtuple('AppendArgs', append_arg_list, defaults=defaults)


class Writer(object):
    """
    An abstract base for specific writers to write the contents
    of a NetCDF file to a different format.

    """
    def __init__(self, data_model,
                 array_filepath=None, container=None, array_name=None,
                 unlimited_dims=None, ctx=None):
        self._data_model = data_model
        self.array_filepath = array_filepath
        self.container = container  # Azure container name.
        self.unlimited_dims = unlimited_dims
        self.ctx = ctx  # TileDB Context object.

        self._scalar_unlimited = None

        # Need either a local filepath or a remote container.
        utils.ensure_filepath_or_container(self.array_filepath, self.container)

        self._array_name = array_name
        if self._array_name is None:
            self.array_name = os.path.basename(os.path.splitext(self.data_model.netcdf_filename)[0])
        else:
            self.array_name = self._array_name
        self.array_path = utils.filepath_generator(self.array_filepath,
                                                   self.container,
                                                   self.array_name,
                                                   ctx=self.ctx)

    @property
    def data_model(self):
        if not self._data_model.dataset_open():
            self._data_model.open()
        return self._data_model

    def _all_coords(self, variable):
        dim_coords = list(variable.dimensions)
        other_coords = variable.coordinates.split(' ')
        return dim_coords + other_coords

    def _get_dim_coord_names(self, var_name):
        """
        Figure out names of the dimension-describing coordinates for this array,
        including the promoted append-dimension scalar coordinate if necessary.

        """
        dim_coord_names = self.data_model.variables[var_name].dimensions
        if self._scalar_unlimited is not None:
            # `dim_coord_names` is a tuple...
            dim_coord_names = (self._scalar_unlimited,) + dim_coord_names
        return dim_coord_names

    def _append_checker(self, other_data_model, var_name, append_dim):
        """Checks to see if an append operation can go ahead."""
        # Sanity checks: is the var name in both self, other, and the tiledb?
        assert var_name in self.data_model.data_var_names, \
            f'Variable name {var_name!r} not found in this data model.'
        assert var_name in other_data_model.data_var_names, \
            f'Variable name {var_name!r} not found in other data model.'

        self_var = self.data_model.variables[var_name]
        self_var_coords = self._all_coords(self_var)
        # Is the append dimension valid?
        assert append_dim in self_var_coords, \
            f'Dimension {append_dim!r} not found in this data model.'

        with other_data_model.open_netcdf():
            other_var = other_data_model.variables[var_name]
            other_var_coords = self._all_coords(other_var)
            # Is the append dimension valid?
            assert append_dim in other_var_coords, \
                f'Dimension {append_dim!r} not found in other data model.'

    def _append_dimension(self, var_name, append_desc):
        """Determine the name and index of the dimension for the append operation."""
        if not isinstance(append_desc, int):
            # Find the append axis from the dimension name.
            append_axis = self.data_model.variables[var_name].dimensions.index(append_desc)
            append_dim = append_desc
        else:
            # Find the append dimension name from the axis.
            append_axis = append_desc
            append_dim = self.data_model.dimensions[append_axis]
        return append_axis, append_dim

    def _fill_missing_points(self, coord_array_path, coord_array_name, verbose=False):
        """
        If one or more indices along the append axis are missing spatial points, we
        end up with `NaN`s in the resultant coordinate array. This prevents loading
        into Iris (as we cannot make a monotonic coordinate array).

        Fill such missing points with interpolated point values so that Iris can load
        the dataset, if with missing data points still. Use a simple custom 1D
        interpolator as the SciPy and NumPy offerings cannot handle NaN values.

        """
        with tiledb.open(coord_array_path, 'r', ctx=self.ctx) as D:
            ned = D.nonempty_domain()[0]
            coord_points = D[ned[0]:ned[1]][coord_array_name]

        missing_points, = np.nonzero(np.isnan(coord_points))
        if len(missing_points):
            if verbose:
                print(f'{len(missing_points)} points to fill in {coord_array_name!r}.')

            ind_points = np.arange(len(coord_points))
            coord_steps = np.unique(np.diff(coord_points))
            # Expects only a single non-NaN step (i.e. monotonicity).
            numeric_step, = coord_steps[np.nonzero(~np.isnan(coord_steps))]

            # Interpolate to fill the missing points.
            vec_interp = np.vectorize(fillnan)
            coord_points[missing_points] = vec_interp(ind_points[missing_points],
                                                      coord_points[0],
                                                      numeric_step)

            # Write the whole filled array back to the TileDB coord array.
            with tiledb.open(coord_array_path, 'w', ctx=self.ctx) as D:
                D[ned[0]:ned[1]] = coord_points
        else:
            if verbose:
                print(f'No missing points in {coord_array_name!r}, nothing to do.')


###################################################################################
#                                                                                 #
# Remove these functions from `TDBWriter` because most of them are static and it  #
# might make tiling in parallel possible!                                         #
#                                                                                 #
###################################################################################


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
    if start_index is None:
        start_index = 0
        shape = data_vars[0].shape  # All data vars *must* have the same shape for writing...
        if scalar:
            shape = (1,) + shape
        write_indices = _array_indices(shape, start_index)
    else:
        write_indices = start_index

    # Write netcdf data var contents into array.
    with tiledb.open(array_filename, 'w', ctx=ctx) as A:
        A[write_indices] = {data_var.name: data_var[...] for data_var in data_vars}


def _dim_inds(dim_points, spatial_inds, offset=0):
    """Convert coordinate values to index space."""
    return [list(dim_points).index(si) + offset for si in spatial_inds]


def _dim_points(points):
    """Convert a dimension variable (coordinate) points to index space."""
    start, stop = points[0], points[-1]
    step, = np.unique(np.diff(points))
    return start, stop, step


def _dim_offsets(dim_points, self_stop_ind, self_stop, self_step,
                 scalar=False, points_offset=None):
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

    if points_offset is None:
        points_offset = other_start - self_stop
    inds_offset = int(points_offset / self_step) + self_stop_ind

    i_start, _ = _dim_inds(dim_points, spatial_inds, inds_offset)
    return i_start


def fillnan(xi, y0, diff):
    """A simple linear 1D interpolator."""
    return y0 + (xi * diff)


def _progress_report(other_data_model, verbose, i, total):
    """A helpful printout of append progress."""
    # XXX Not sure about this when called from a bag...
    if verbose and i is not None and total is not None:
        fn = other_data_model.netcdf_filename
        ct = i + 1
        pc = 100 * (ct / total)
        print(f'Processing {fn}...  ({ct}/{total}, {pc:0.1f}%)', end="\r")


def _make_multiattr_tile(other_data_model, domain_path, data_array_name,
                         var_names, append_axis, append_dim, scalar_coord,
                         self_ind_stop, self_dim_stop, self_step,
                         scalar_offset=None, do_logging=False, ctx=None):
    """Process appending a single tile to `self`, per domain."""
    other_data_vars = [other_data_model.variables[var_name] for var_name in var_names]
    data_var_shape  = other_data_vars[0].shape
    other_dim_var = other_data_model.variables[append_dim]
    other_dim_points = np.atleast_1d(other_dim_var[:])

    # Check for the dataset being scalar on the append dimension.
    if not scalar_coord and len(other_dim_points) == 1:
        scalar_coord = True

    if scalar_coord:
        shape = [1] + list(data_var_shape)
    else:
        shape = data_var_shape

    offsets = []
    offset = _dim_offsets(
        other_dim_points, self_ind_stop, self_dim_stop, self_step,
        scalar=scalar_coord, points_offset=scalar_offset)
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

    do_logging = False
    if job_args.logfile is not None:
        do_logging = True
        logging.basicConfig(filename=job_args.logfile,
                            level=logging.ERROR,
                            format='%(asctime)s %(message)s',
                            datefmt='%d/%m/%Y %H:%M:%S')

    domains_mapping = job_args.mapping
    domain_paths = job_args.domain
    append_dim = job_args.dim
    append_axes = job_args.axis

    # Record what we've processed...
    if do_logging:
        logging.error(f'Processing {job_args.other!r} ({job_args.job_number+1}/{job_args.n_jobs})')

    # To improve fault tolerance all the append processing happens in a try/except...
    try:
        if isinstance(job_args.other, NCDataModel):
            other_data_model = job_args.other
        else:
            other_data_model = NCDataModel(job_args.other)

        with other_data_model.open_netcdf():
            for n, domain_path in enumerate(domain_paths):
                if job_args.verbose:
                    fn = other_data_model.netcdf_filename
                    job_no = job_args.job_number
                    n_jobs = job_args.n_jobs
                    n_domains = len(domain_paths)
                    print(f'Processing {fn}...  ({job_no+1}/{n_jobs}, domain {n+1}/{n_domains})', end="\r")

                append_axis = append_axes[n]
                if domain_path.endswith('/'):
                    _, domain_name = os.path.split(domain_path[:-1])
                else:
                    _, domain_name = os.path.split(domain_path)
                array_var_names = domains_mapping[domain_name]
                _make_multiattr_tile(other_data_model, domain_path, job_args.name,
                                     array_var_names, append_axis, append_dim, job_args.scalar,
                                     job_args.ind_stop, job_args.dim_stop, job_args.step,
                                     scalar_offset=job_args.offset, do_logging=do_logging, ctx=ctx)
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
