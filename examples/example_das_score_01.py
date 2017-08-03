#!/usr/bin/env python3
import enstools
import numpy
import matplotlib.pyplot as plt


def create_example_data(offset):
    """
    create example forecast and observation

    Parameters
    ----------
    offset : float
            value between 0 and 8

    Returns
    -------
    numpy.ndarray, (1000, 1000)
    """
    x = numpy.linspace(-8 + offset, 8 + offset, 1000)
    y = numpy.linspace(-4, 4, 1000)
    xv, yv = numpy.meshgrid(x, y)
    data = xv * numpy.exp(-xv ** 2 - yv ** 2) * 10
    data = numpy.where(data > 0.5, data, 0)
    return data


if __name__ == "__main__":
    # generate data
    obs = create_example_data(0.0)
    fct = create_example_data(-1.0)

    # calculate das score
    das = enstools.scores.das(obs, fct, factor=5, threshold=0)
    print('Score: DAS %.2f, DIS %.2f, AMP %.2f' % (das["das"], das["dis"], das["amp"]))

    # average displacement vectors for plotting
    xdis_o = enstools.interpolation.downsize(das["xdis_o"], 50)
    ydis_o = enstools.interpolation.downsize(das["ydis_o"], 50)

    # create a plot comparable to Keil and Craig 2009, Fig. 2.
    fig, ax = plt.subplots(2, 2, subplot_kw={"ylim": [0, 1000], "xlim": [0, 1000], "aspect": "equal"})
    for r in [0, 1]:
        for c in [0, 1]:
            ax[r, c].plot([0, 1000], [500, 500], "--", color="lightgray")
            ax[r, c].plot([500, 500], [0, 1000], "--", color="lightgray")
    ax[0, 0].imshow(obs, cmap="Greys")
    ax[0, 0].text(50, 950, "Observation", verticalalignment="top", fontsize=14)
    ax[0, 1].imshow(fct, cmap="Greys")
    ax[0, 1].text(50, 950, "Forecast", verticalalignment="top", fontsize=14)
    ax[1, 0].imshow(fct, cmap="Greys")
    ax[1, 0].quiver(numpy.linspace(25, 975, 20), numpy.linspace(25, 975, 20), xdis_o, ydis_o, scale=1000)
    ax[1, 0].text(50, 950, "Forecast and\nDisplacement-Vectors", verticalalignment="top", fontsize=14)
    ax[1, 1].imshow(das["morph_o"], cmap="Greys")
    ax[1, 1].text(50, 950, "Forecast morphed to\nObservations", verticalalignment="top", fontsize=14)
    plt.setp([a.get_xticklabels() for a in ax[0, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in ax[:, 1]], visible=False)
    plt.tight_layout()
    plt.show()
