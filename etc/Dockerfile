FROM python:3.7

# Enable discovery of PROJ shared objects files
ENV LD_LIBRARY_PATH /lib:/usr/lib:/usr/local/lib

# Install PROJ - dependency for cartopy (a dependency of scitools-iris)
RUN apt-get update && \
    apt-get install -y sqlite3 && \
    wget https://download.osgeo.org/proj/proj-7.0.0.tar.gz && \
    tar -xf proj-7.0.0.tar.gz && \
    cd proj-7.0.0 && \
    ./configure && \
    make && \
    make install && \
    projsync --system-directory --all && \
    cd ..

# Install GEOS - dependency for cartopy (a dependency of scitools-iris)
# RUN apt-get install -y libgeos-dev
RUN wget http://download.osgeo.org/geos/geos-3.8.1.tar.bz2 && \
    tar -xf geos-3.8.1.tar.bz2 && \
    cd geos-3.8.1 && \
    ./configure && \
    make && \
    make install && \
    cd ..

# scitools-iris has a dependency on cf_units, which is itself a wrapper around the UDUNITS-2 C library. Install UDUNITS-2 library below.
RUN wget ftp://ftp.unidata.ucar.edu/pub/udunits/udunits-2.2.26.tar.gz && \
    tar -xf udunits-2.2.26.tar.gz && \
    cd udunits-2.2.26 && \
    ./configure && \
    make && \
    make install && \
    cd ..

# scitools-iris has a dependency on pyke, which is not available through pypi. pyke be installed manually.
RUN wget https://sourceforge.net/projects/pyke/files/pyke/1.1.1/pyke3-1.1.1.zip/download -O pyke3-1.1.1.zip && \
    unzip pyke3-1.1.1.zip && \
    cd pyke-1.1.1 && \
    python setup.py install && \
    cd ..

# Must have an available numpy installation (1.10+) prior to installing cartopy during setup.py
RUN pip install numpy==1.18.5

WORKDIR /tiledb_netcdf

# Copy only files needed to install requirements to speed up subsequent builds
COPY setup.py README.md .
RUN pip install .

COPY . /tiledb_netcdf

ENTRYPOINT ["python"]
