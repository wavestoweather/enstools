#!/bin/bash
set -e

# run all tests with python2 and python3
nosetests --nocapture --nologcapture --with-doctest
nosetests3 --nocapture --nologcapture --with-doctest
