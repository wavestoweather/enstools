# Load virtual environment

# create a virtual environement and install all dependencies
if [[ ! -d venv ]] ; then
    python3 -m venv --prompt enstools venv
    source venv/bin/activate
    pip install -U pip
    # TODO: This workaround should allow us to install the proper dependencies even with the latest version of pip.
    pip install cartopy==0.19.0.post1

    pip install wheel
    pip install -e .
    pip install --force-reinstall pytest
fi

source venv/bin/activate

set -xuve

# Enter examples folder:
cd examples


# List of examples to run

# The following example fails from time to time
# 
# example_download_radar.py
#
# Until that is fixed it won't be executed by this script.

# Run working examples

# Example plot Icon 01
python example_plot_icon_01.py --save example_plot_icon_01.png


# Example das score 01
python example_das_score_01.py
