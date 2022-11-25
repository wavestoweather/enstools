Scores
======

Enstools includes several functions to compare data:

:meth:`enstools.scores.das`

:meth:`enstools.scores.mean_square_error`

:meth:`enstools.scores.root_mean_square_error`

:meth:`enstools.scores.normalized_root_mean_square_error`

:meth:`enstools.scores.pearson_correlation`

:meth:`enstools.scores.structural_similarity_index`

:meth:`enstools.scores.peak_signal_to_noise_ratio`

:meth:`enstools.scores.kolmogorov_smirnov`

.. code::

    from enstools.scores import pearson_correlation

    corr = pearson_correlation(reference, target)


See API
-------

:ref:`scores-api`.
