# source this file!
if [[ "${BASH_SOURCE[0]}" == "${0}" ]] ; then
    echo "ERROR: use: source ${BASH_SOURCE[0]}"
    exit 1
fi

# load required modules
source venv-functions.sh
setup_environment

# activate python environment
source venv/bin/activate
export HDF5_DISABLE_VERSION_CHECK=2
