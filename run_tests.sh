#!/bin/bash
set -e

function usage {
    echo "arguments:"
    echo "-r    skip tests with R"
    exit -1
}

# parse the command line
excluded_files=""
skip_python2=false
skip_python3=false
while getopts "rh" opt ; do
    case $opt in
        r)
            echo "INFO: not running tests with R!"
            excluded_files="tests/test_scores_scoringRules_01.py"
            ;;
        h)
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
    pip install -e .
    pip install --force-reinstall pytest
fi

source venv/bin/activate
pytest ${ignore_option}
