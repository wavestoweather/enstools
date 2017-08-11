#!/bin/bash
set -e

function usage {
    echo "arguments:"
    echo "-r    skip tests with R"
    exit -1
}

# parse the command line
excluded_files=""
while getopts "rh" opt ; do
    case $opt in
        r)
            excluded_files=test_scores_scoringRules_01.py
            ;;
        h)
            usage
            ;;
    esac
done
if [[ ! -z $excluded_files ]] ; then
    ignore_option="--ignore-file=$excluded_files"
fi

# run all tests with python2 and python3
echo "#################################################################################################################"
echo "Running all tests with python 2 ...."
echo "#################################################################################################################"
echo
nosetests --nocapture --nologcapture --with-doctest $ignore_option

echo
echo "#################################################################################################################"
echo "Running all tests with python 3 ...."
echo "#################################################################################################################"
echo
nosetests3 --nocapture --nologcapture --with-doctest $ignore_option
