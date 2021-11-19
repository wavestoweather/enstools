# Ensemble Tools

This package provides core functionality to Python-based tools developed within
the framework of [Waves to Weather - Transregional Collaborative Research 
Project (SFB/TRR165)](https://wavestoweather.de). 

Shared functionality includes:
- Clustering (`enstools.clustering`)
- Interpolation (`enstools.interpolation`)
- Reading and Writing data (`enstools.io`)
- Retrieval of open data (`enstools.opendata`)
- Post-processing (`enstools.post`)
- Scores (`enstools.scores`)

# Installation using pip

`pip` is the easiest way to install `enstools` along with all dependencies. It
is recommended and not necessary to do that in a separate virtual environment. 

## Preparation of a local environment

The steps outlined here can be done inside of a working-copy of this,
repository. The created directory `venv` will be ignored by git.

At first create a new python virtual environment:

    python3 -m venv --prompt=enstools venv

That will create a new folder `venv` containing the new environment. To use
this environment, we need to activate it:

    source venv/bin/activate

Next we
need to update `pip` and install `wheel`. Both are required in up-to-date 
versions for the installation to run:

    pip install --upgrade pip wheel

## Installation

For development, you can create a clone of this repository and install that
local copy in development mode into your virtual environment. This is 
especially useful if you plan to edit the code of `enstools`. Python scripts
using the virtual environment will immediately see all your changes with the
need to reinstall anything.

    git clone https://github.com/wavestoweather/enstools.git
    cd enstools
    pip install -e .

If you have no plans to modify any code, then you can install `enstools`
without creating a local working-copy before:

    pip install git+https://github.com/wavestoweather/enstools.git

# Acknowledgment and license

Ensemble Tools (`enstools`) is a collaborative development within
Waves to Weather (SFB/TRR165) coordinated by the subproject 
[Z2](https://www.wavestoweather.de/research_areas/phase2/z2) and funded by the
German Research Foundation (DFG).

A full list of code contributors can [CONTRIBUTORS.md](./CONTRIBUTORS.md).

The code is released under an [Apache-2.0 licence](./LICENSE).
