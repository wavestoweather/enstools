# use an eccodes version compiled with GIL release

# How to compile:
#   cd python
#   $SCRATCH/swig/swig-1.3.40-install/bin/swig -python -threads -module gribapi_swig -o swig_wrap_numpy.c gribapi_swig.i
#   cp gribapi_swig.py swig_wrap_numpy.py
#   cd ../../build
#   cmake ../eccodes-2.4.1-Source -DCMAKE_INSTALL_PREFIX=/project/meteo/scratch/Robert.Redl/eccodes/install-2.4.1-threads -DENABLE_ECCODES_THREADS=ON
#   make
#   make install

# Library Paths
prefix=/project/meteo/scratch/Robert.Redl/eccodes/install-2.4.1-threads
export LD_LIBRARY_PATH=${prefix}/lib:$LD_LIBRARY_PATH
export PYTHONPATH=${prefix}/lib/python2.7/site-packages:$PYTHONPATH

# DWD Grib-Definitions
export ECCODES_DEFINITION_PATH=/software/meteo/xenial/x86_64/eccodes/2.4.1-gcc/share/eccodes/definitions.edzw:/software/meteo/xenial/x86_64/eccodes/2.4.1-gcc/share/eccodes/definitions