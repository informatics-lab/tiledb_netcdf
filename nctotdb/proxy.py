from distributed.protocol import dask_serialize, dask_deserialize
import numpy as np
import tiledb


# Inspired by https://github.com/SciTools/iris/blob/master/lib/iris/fileformats/netcdf.py#L418.
class TileDBDataProxy(object):
    """A proxy to the data of a single TileDB array attribute."""

    __slots__ = ("shape", "dtype", "path", "var_name", "ctx", "handle_nan")

    def __init__(self, shape, dtype, path, var_name, ctx=None, handle_nan=None):
        self.shape = shape
        self.dtype = dtype
        self.path = path
        self.var_name = var_name
        self.ctx = ctx
        self.handle_nan = handle_nan

    @property
    def ndim(self):
        return len(self.shape)

    def __getitem__(self, keys):
        with tiledb.open(self.path, 'r', ctx=self.ctx) as A:
            data = A[keys][self.var_name]
            if self.handle_nan is not None:
                if self.handle_nan == 'mask':
                    data = np.ma.masked_invalid(data, np.nan)
                elif type(self.handle_nan) in [int, float]:
                    data = np.nan_to_num(data, nan=self.handle_nan, copy=False)
                else:
                    raise ValueError(f'Not a valid nan-handling approach: {self.handle_nan!r}.')
        return data

    def serialize_state(self, dummy=None):
        """
        Take the current state of `self` and make it serializable.

        Note the apparently unused kwarg `dummy`. This allows `serialize_state` to be used
        as the 'default' serialization function for msgpack. For example:

        ```
        msgpack.dumps(my_data_proxy, default=my_data_proxy.serialize_state)
        ```

        In such instances, msgpack calls `default` with the object to be dumped, which makes
        no sense in this application.

        """
        state = {}
        for attr in self.__slots__:
            value = getattr(self, attr)
            if attr == "shape":
                state[attr] = [int(i) for i in value]
            elif attr == "dtype":
                state[attr] = np.dtype(value).str
            elif attr == "ctx":
                # ctx is based on a C library that doesn't pickle...
                state[attr] = value.config().dict() if value is not None else None
            else:
                state[attr] = value
        return state

    def __getstate__(self):
        """Simplify a complex object for pickling."""
        return self.serialize_state()

    def __setstate__(self, state):
        """Restore the complex object from the simple pickled dict."""
        deserialized_state = deserialize_state(state)
        for key, value in deserialized_state.items():
            if key in self.__slots__:
                setattr(self, key, value)


def deserialize_state(s_state):
    """
    Take a serialized dictionary of state and deserialize it to set state
    on a `TileDBDataProxy` instance.

    """
    d_state = {}
    for key, s_value in s_state.items():
        if key == "dtype":
            d_value = np.dtype(s_value)
        elif key == "ctx":
            d_value = tiledb.Ctx(config=tiledb.Config(s_value)) if s_value is not None else None
        else:
            d_value = s_value
        d_state[key] = d_value
    return d_state
