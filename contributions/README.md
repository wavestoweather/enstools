# General Guidelines for contribution to Ensemble Tools
## Create a namespace package

Contributions are made as so-called namespace packages. The namespace in our case is `enstools`, new contributions will 
accordingly have names like `enstools.contribution`. To get started, a new git repository with the content of 
`enstools-dummy` can be created.


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
    
... to be extended!
