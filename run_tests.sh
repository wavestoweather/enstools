#!/bin/bash
set -e

function usage {
    echo "arguments:"
    echo "-w    warnings are errors"
    exit 1
}

# parse the command line
excluded_files=""
extra_arguments=""
while getopts "w" opt ; do
    case $opt in
        w)
            echo "WARNING: warnings are treated like errors for debugging."
            extra_arguments="-W error"
            ;;
        *)
            usage
            ;;
    esac
done
if [[ ! -z $excluded_files ]] ; then
    ignore_option="--ignore=$excluded_files"
fi


# create a virtual environement and install all dependencies
if [[ ! -d venv ]] ; then
    python3 -m venv --prompt enstools venv
    source venv/bin/activate
    pip install -U pip
    pip install wheel

    # In ubuntu 22.04 proj >8.0 is available which allows us to install cartopy >= 0.20 but not
    # Previous versions
    # We'll check ubuntu's version and in case its previous to 22.04 we'll preinstall cartopy 0.19
    source /etc/os-release
    if (( $(echo "${VERSION_ID} < 22.04" |bc -l) )); then
        pip install cartopy==0.19.0.post1
    fi
	    
    pip install -e .
    pip install --force-reinstall pytest
fi

source venv/bin/activate
pytest ${extra_arguments} ${ignore_option}
