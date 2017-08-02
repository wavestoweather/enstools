from enstools.core import check_arguments
from .calc_das import threshold_data
from .match_pyramid_ic import match_pyramid
import numpy as np


@check_arguments(shape={"obs": (0, 0), "fct": (0, 0)})
def das(obs, fct, factor=4, threshold=None):
    """
    calculate DAS - Displacement and amplitude score for observation and forecast

    based on IDL programm Version 2.0 20090918 C. Keil (DLR) , G.C. Craig (DLR), H. Mannstein (DLR)
    and python scripts from Bettina Richter (Meteorologisches Institut - LMU Muenchen)

    DAS is a displacement-based error measure and includes
    two components: displacement and amplitude error, where
      - displacement is scaled with maximum displacement c1
      - amplitude error is scaled with 'climatological' observed mean c2

    'Philosophy': complete miss equals 100% amplitude error
    The smaller DAS the better the forecast quality.

    das = dis_nor + lse_nor

    Here both components have equal weight. This is a slight modification
    to the application of FQM on satellite imagery taking the maximum of
    both components published in the reference publication


    Parameters
    ----------
    obs : xarray.DataArray or np.ndarray
            2d array with observations

    fct : xarray.DataArray or np.ndarray
            2d array with forecasts

    factor: int
            sub-sampling factor or topmost pyramid level (default: 4)

    threshold : float
            threshold of data (default: None)

    Returns
    -------
    dict
            The score itself and all related values:
            das:        the DAS score
            dis:        the displacement component
            amp:        the amplitude component
            rms_obs:    root mean square of observation
            morph_o:    morphed fct to obs
            morph_f:    morphed obs to fct
            xdis_o:     x-displacement of fct morphed to obs
            ydis_o:     y-displacement of fct morphed to obs
            xdis_f:     x-displacement of obs morphed to fct
            ydis_f:     y-displacement of obs morphed to fct
            lse_o:      least square error of fct morphed to obs
            lse_f:      least square error of fct morphed to obs


    References
    ----------
    .. [1] Keil C, Craig G.C., 2009: A displacement and amplitude score
       employing an optical flow technique. WAF, doi:10.1175/2009WAF2222247.1

    .. [2] Keil C, Craig G.C., 2007: A displacement-based error measure applied
       in a regional ensemble forecasting system. MWR 135, 3248-3259, doi:10.1175/MWR3457.1
    """

    # apply threshold
    if threshold is not None:
        obs = threshold_data(obs, threshold)
        fct = threshold_data(fct, threshold)

    # morph fct to obs, obs-space
    morph_o, xdis_o, ydis_o, lse_o = match_pyramid(fct, obs, factor=factor)
    # morph obs to fct, fct-space
    morph_f, xdis_f, ydis_f, lse_f = match_pyramid(obs, fct, factor=factor)

    # calculate DAS - displacement and almplitude score
    dis_max = np.sqrt(2.) * 2. ** (factor + 2.)
    x1 = np.int(2 * 2 ** factor)
    y2, x2 = [np.int(s - x1) for s in np.shape(obs)]
    das, dis, amp, rms_obs = calc_das.calc_das(obs, fct, xdis_o, ydis_o, lse_o, xdis_f, ydis_f, lse_f, dis_max,
                                               threshold=threshold, x1=x1, y1=x1, x2=x2, y2=y2)

    # construct the result
    result = {"das": das, "dis": dis, "amp": amp, "rms_obs": rms_obs, "morph_o": morph_o, "morph_f": morph_f,
              "xdis_o": xdis_o, "xdis_f": xdis_f, "ydis_o": ydis_o, "ydis_f": ydis_f, "lse_o": lse_o, "lse_f": lse_f}
    return result
