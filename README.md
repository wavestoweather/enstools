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

## Installation using pip in local environment

At first create a new python virtual environment:

    python3 -m venv --prompt=enstools venv

That will create a new folder `venv` containing the new environment. To use
this environment, we need to activate it:

    source venv/bin/activate

Next we
need to update `pip` and install `wheel`. Both are required in up-to-date 
versions for the installation to run:

    pip install --upgrade pip wheel

Now we can install `enstools` in development mode into our new environment:

    pip install -e .
