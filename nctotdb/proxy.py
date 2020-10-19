from distributed.protocol.numpy import serialize_numpy_maskedarray, deserialize_numpy_maskedarray
import numpy as np
import tiledb


# Inspired by https://github.com/SciTools/iris/blob/master/lib/iris/fileformats/netcdf.py#L418.
class TileDBDataProxy:
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

    def __getstate__(self):
        return {attr: getattr(self, attr) for attr in self.__slots__}

    def __setstate__(self, state):
        for key, value in state.items():
            setattr(self, key, value)


def tdb_data_proxy_dumps(data_proxy):
    header = {"shape": data_proxy.shape,
              "dtype": data_proxy.dtype,
              "path": data_proxy.path,
              "var_name": data_proxy.var_name,
              "handle_nan": data_proxy.handle_nan,
              "ctx": data_proxy.ctx.config().dict() if data_proxy.ctx is not None else None}
    return header, []


def tdb_data_proxy_loads(header, frames):
    if header["ctx"] is not None:
        ctx = tiledb.Ctx(config=tiledb.Config(header["ctx"]))
    else:
        ctx = None
    result = TileDBDataProxy(header["shape"],
                             header["dtype"],
                             header["path"],
                             header["var_name"],
                             ctx=ctx,
                             handle_nan=header["handle_nan"])
    return result