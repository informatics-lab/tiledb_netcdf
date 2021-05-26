import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tiledb_netcdf",
    version="0.3.0",
    author="Peter Killick",
    author_email="peter.killick@informaticslab.co.uk",
    description="An adapter to convert NetCDF files to TileDB or Zarr arrays",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/informatics-lab/tiledb_netcdf",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "distributed>=2.28.0",
        "tiledb>=0.6.6",
        "scitools-iris>=2.4.0",
        "xarray>=0.15.1",
        "zarr>=2.4.0"
    ]
)
