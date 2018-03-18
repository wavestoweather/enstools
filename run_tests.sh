#!/bin/bash
set -e

function usage {
    echo "arguments:"
    echo "-r    skip tests with R"
    echo "-3    run only python3 tests"
    echo "-2    run only python2 tests"
    exit -1
}

# parse the command line
excluded_files=""
skip_python2=false
skip_python3=false
while getopts "rh23" opt ; do
    case $opt in
        r)
            excluded_files=test_scores_scoringRules_01.py
            ;;
        2)
            skip_python3=true
            ;;
        3)
            skip_python2=true
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
if [[ $skip_python2 == "false" ]] ; then
    echo "#################################################################################################################"
    echo "Running all tests with python 2 ...."
    echo "#################################################################################################################"
    echo
    nosetests --nocapture --nologcapture --with-doctest $ignore_option
fi

if [[ $skip_python3 == "false" ]] ; then
    echo
    echo "#################################################################################################################"
    echo "Running all tests with python 3 ...."
    echo "#################################################################################################################"
    echo
    nosetests3 --nocapture --nologcapture --with-doctest $ignore_option
fi