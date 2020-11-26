#!/bin/bash -l
# start load required modules and start the python kernel.
# The first argument for this script must be the connection file

# find folder of installation
BASEDIR=$(dirname "${BASH_SOURCE[0]}")

# source setup script
cd $BASEDIR
source venv-activate.sh

# numba should use all cores
export NUMBA_NUM_THREADS=$(nproc)

# the kernel needs to find this folder for importing packages
export PYTHONPATH=$BASEDIR:$PYTHONPATH

# start the actual python kernel
python -m ipykernel_launcher -f "${1}"


