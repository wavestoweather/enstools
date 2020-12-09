# General Guidelines for contribution to Ensemble Tools
## Create a namespace package

Contributions are made as so-called namespace packages. The namespace in our case is `enstools`, new contributions will 
accordingly have names like `enstools.contribution`. 


## Setup for development environment

* To get started, a new git repository with the content of `enstools-dummy` can be created.
* The `enstools-dummy` example contains all required parts for a namespace package.
* First step: find a good name for your package and write it to `package.conf`.
* Rename the folder `enstools/dummy` to your new package name.
* Execute `venv-setup.sh`, that will do the following:
    * create a new virtual environment in a new folder `venv`.
    * install all dependencies including the `enstools`-core Package. The core package is installed editable, which 
    should make required additions and modifications easier. The source code is installed here: `venv/src/enstools`
    * Install a Jupyter-Kernel for the new environment. If done at LMU, it will be available in our 
    Jupyterhub.
    * the new namespace package in installed editable into the virtual environment.


## Exchange between functions

* Where appropriate, new functions should take `xarray` objects as arguments. Such objects are for example returned by
`enstools.io.read`. 
* Variables:
    * names should follow the CF-convention (https://cfconventions.org).
    * variables should have a `units` attributes containing SI-units.
    * units can by converted automatically using the decorator `@check_arguments` 
    (see `convective_adjustment_time_scale`)
* Dimensions:
    * `read` returns this order: `(time, ens, layer, cells)`, `cells` could also be `lat`, `lon`. Where possible, 
    it should be used. 
    * names should as for variables follow CF-conventions.
    * some support functions for dimensions are available in `enstools.misc`, 
    e.g., `get_ensemble_dim`, `get_time_dim`, ...
    

## Documentation and source code format

* A README.md file in the root folder of the repository is required. 
* All other documentation should be included in the source code files.
* Format for documentation: Numpy-Docstring-Style (https://numpydoc.readthedocs.io/en/latest/format.html).
* Documentation-Webpages will be created with Sphinx (see core package for reference).
* Source code files should be formated according to the pep8 (https://www.python.org/dev/peps/pep-0008/) standard. 


... to be extended!
