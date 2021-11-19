import numpy as np
from enstools.core import check_arguments


def threshold_data(arr, threshold):
    """
    set elements of arr to zero if <= thres

    Parameters
    ----------
    arr : xarray.DataArray or np.ndarray
            input array

    threshold : float
            threshold below which values should be set to zero

    Returns
    -------

    """
    # set elements of arr to zero if <= thres
    return np.where(arr <= threshold, 0, arr)


@check_arguments(shape={"obs": (0, 0),
                        "fct": "obs",
                        "xd_o": "obs",
                        "yd_o": "obs",
                        "lse_o": "obs",
                        "xd_f": "obs",
                        "yd_f": "obs",
                        "lse_f": "obs"})
def calc_das(obs, fct,
             xd_o, yd_o, lse_o,
             xd_f, yd_f, lse_f,
             dis_max,
             threshold=1., x1=0, y1=0, x2=0, y2=0):
    """
    Calculate DAS (Displacement and Amplitude Score)

    Parameters
    ----------
    obs : xarray.DataArray or np.ndarray
            observation

    fct : xarray.DataArray or np.ndarray
            forecast

    xd_o : xarray.DataArray or np.ndarray
            x-displacement in observation space

    yd_o : xarray.DataArray or np.ndarray
            y-displacement in observation space

    lse_o : xarray.DataArray or np.ndarray
            least-square errors in observation space

    xd_f : xarray.DataArray or np.ndarray
            x-displacement in forecast space

    yd_f : xarray.DataArray or np.ndarray
            y-displacement in forecast space

    lse_f : xarray.DataArray or np.ndarray
            least-square errors in forecast space

    dis_max : float
            maximum search distance

    threshold : float
            threshold value (default: 1.)

    x1 : int
            cut of region, left end (default: 0)

    y1 : int
            cut of region, upper end (default: 0)

    x2 : int
            cut of region, right end (default: 0)

    y2 : int
            cut of region, lower end (default: 0)

    Returns
    -------
    tuple:
            das, dis, amp, rms_obs
    """

    # cut off ends
    if x1 or x2 or y1 or y2:
        obs = obs[y1:y2, x1:x2]
        fct = fct[y1:y2, x1:x2]
        xd_o = xd_o[y1:y2, x1:x2]
        yd_o = yd_o[y1:y2, x1:x2]
        lse_o = lse_o[y1:y2, x1:x2]
        xd_f = xd_f[y1:y2, x1:x2]
        yd_f = yd_f[y1:y2, x1:x2]
        lse_f = lse_f[y1:y2, x1:x2]

    dis_o = np.sqrt(xd_o ** 2 + yd_o ** 2)
    dis_f = np.sqrt(xd_f ** 2 + yd_f ** 2)

    if threshold is not None:
        cond_o = np.where((obs > threshold))
        cond_f = np.where((fct > threshold))
    else:
        cond_o = np.empty(obs.shape, dtype=bool)
        cond_o[:] = True
        cond_f = cond_o
    n_obs = np.size(obs[cond_o])
    n_fct = np.size(fct[cond_f])

    dis_o_mean = amp_o_mean = dis_f_mean = amp_f_mean = rms_obs = 0.

    if n_obs > 0:
        rms_obs = np.sqrt(np.mean(obs[cond_o] ** 2))  # root mean square of observation
        dis_o_mean = np.mean(dis_o[cond_o])
        amp_o_mean = np.sqrt(np.mean(lse_o[cond_o]))  # root mean square error
    if n_fct > 0:
        dis_f_mean = np.mean(dis_f[cond_f])
        amp_f_mean = np.sqrt(np.mean(lse_f[cond_f]))  # root mean square error

    if n_obs or n_fct:
        # Displacement Error
        dis = (n_obs * dis_o_mean + n_fct * dis_f_mean) / (n_obs + n_fct)
        # Amplitude Error
        amp = (n_obs * amp_o_mean + n_fct * amp_f_mean) / (n_obs + n_fct)

        # DAS - Displacement and amplitude score
        das = dis / dis_max + amp / rms_obs

        return das, dis / dis_max, amp / rms_obs, rms_obs

    else:
        return np.nan, np.nan, np.nan, np.nan
