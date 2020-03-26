import os


class ArrayPath(object):
    def construct_path(self, group_name, array_name):
        raise NotImplementedError


class PosixArrayPath(ArrayPath):
    def __init__(self, base_path, tiledb_name):
        self.base_path = base_path
        self.tiledb_name = tiledb_name

        self.basename = os.path.join(self.base_path, self.tiledb_name)

    def construct_path(self, group_name, array_name):
        return os.path.join(self.basename, group_name, array_name)


class AzureArrayPath(ArrayPath):
    def __init__(self, container, tiledb_name, ctx):
        self.container = container
        self.tiledb_name = tiledb_name
        self.ctx = ctx

        self.basename = f'azure://{self.container}/{self.tiledb_name}'

    def construct_path(self, group_name, array_name):
        return f'{self.basename}/{group_name}/{array_name}'


class AWSArrayPath(ArrayPath):
    def __init__(self):
        raise NotImplementedError