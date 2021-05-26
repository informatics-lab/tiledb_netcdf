"""Utility functions for `tiledb_netcdf`."""


from .paths import PosixArrayPath, AzureArrayPath


def ensure_filepath_or_container(filepath, container):
    """Ensure that either a filepath or a container has been specified, but not both."""
    # Need either a local filepath or a remote container.
    if filepath is None and container is None:
        raise ValueError("Must supply one of: array filepath, azure container.")
    if filepath is not None and container is not None:
        raise ValueError("Must supply either: array filepath; azure container, but got both.")


def filepath_generator(array_filepath, container, array_name, ctx=None):
    result = None
    if array_filepath is not None:
        result = PosixArrayPath(array_filepath, array_name)
    elif container is not None:
        result = AzureArrayPath(container, array_name, ctx=ctx)
    return result


def fillnan(xi, y0, diff):
    """A simple linear 1D interpolator."""
    return y0 + (xi * diff)
