"""
This module contains python scripts to perform several RWP diagnostics.

Authors:   Joachim Eichhorn, Georgios Fragkoulidis, Michael Riemer,
           Ilona Glatt, Gabriel Wolf, Gil Fleger

The scripts have been tested in a Python 3.5 environment using the
anaconda framework. Required modules are listed below.

"""
from pylab import *
from datetime import timedelta
from mpl_toolkits.basemap import Basemap
from scipy import fft, ifft, arange
from scipy.interpolate import griddata, RectBivariateSpline
import xarray
from enstools.plot.core import get_coordinates_from_xarray
from enstools.plot import contour


def wsp_perpendicular(u,v,ubg,vbg):
    res = (v*ubg - u*vbg) / np.sqrt(ubg**2 + vbg**2)
    return res


def envelope_along_streamline(ubg, vbg, wave, lati, longi, fac, alpha):
    """
    Calculate envelope of a wave signal along streamlines of a 2D background
    flow in spherical geometry. The streamlines are computed from the wind
    components ubg, vbg. wave is the amplitude of the wave signal. LON and LAT
    are the arrays of longitude and latitude values of the wind components.
    delta is the spatial increment (in degrees) used in the integration of the
    streamlines. This version assumes that data coverage in longitude covers
    the hemisphere ([0, 360[).
    The method is described in Zimin et al. 2006. The implementation here
    follows the paper very closely.

    EXPECTS THAT ALL FIELDS ARE ORDERED LAT, LON!

    ubg, vbg: 2d-fields of background flow
    wave: wave signal perpendicular to background flow
    lati, longi: spherical coordinates, MUST be equally spaced in both directions
    fac: fator to control resolution of streamline calculation
    alpha: factor to control width of Gaussian filter applied to wave signal

    """
    # preliminaries
    NX = ubg[:,0].size
    NY = ubg[0,:].size

    LON, LAT = np.meshgrid(longi,lati)

    # check consistency of grid resolution
    if abs(lati[0]-lati[1]) != abs(longi[0]-longi[1]):
        raise IOError('There is a problem with the grid resolution')
    res = abs(lati[0]-lati[1])

    # resolution for streamline calculation
    delta = fac*res
    # Calculate number of integration steps (length of integration should be approx. 360 degree)
    N = int(np.ceil(180./delta))

    # Create bivariate spline approximation of the fields
    # Extend fields (and longitude) in longitude by one data point to account for zonal periodicity
    lon_ext = np.concatenate((longi,[360]))
    uRBS = RectBivariateSpline(lati, lon_ext, np.concatenate((ubg[:,:],np.array([ubg[:,-1]]).T),axis=1))
    vRBS = RectBivariateSpline(lati, lon_ext, np.concatenate((vbg[:,:],np.array([vbg[:,-1]]).T),axis=1))
    waveRBS = RectBivariateSpline(lati, lon_ext, np.concatenate((wave,np.array([wave[:,-1]]).T),axis=1))

    # initialize fields for interpolation (need to be 1D arrays)
    wave_int = np.empty((wave.size, 2*N+1))
    wave_int[:,N] = wave.flatten()
    lat_int = np.empty_like(wave_int)
    lon_int = np.empty_like(wave_int)
    #LAT, LON = np.meshgrid(lat,lon)
    lat_int[:,N] = LAT.flatten()
    lon_int[:,N] = LON.flatten()

    """
    Calculate streamlines of [ubg, vbg] and interpolate wave signal (exactly as in Zimin et al. 2006)
    """
    for n in range(N):
        # forward integration
        u_int = uRBS.ev(lat_int[:,N+n],lon_int[:,N+n])
        v_int = vRBS.ev(lat_int[:,N+n],lon_int[:,N+n])
        wsp = np.sqrt(u_int**2+v_int**2)
        if wsp.any == 0.:
            raise Exception('Wind speed equals zero: division by zero')
        lon_int[:,N+n+1] = (lon_int[:,N+n] + (delta/np.cos(np.pi*lat_int[:,N+n]/180.)) * u_int/ wsp) % 360 # modulo 360; yields values between 0 and 360
        lat_int[:,N+n+1] = lat_int[:,N+n] + (delta * v_int/ wsp)
        wave_int[:,N+n+1] = waveRBS.ev(lat_int[:,N+n+1],lon_int[:,N+n+1])

        # backward integration
        u_int = uRBS.ev(lat_int[:,N-n], lat_int[:,N-n])
        v_int = vRBS.ev(lat_int[:,N-n],lat_int[:,N-n])
        wsp = np.sqrt(u_int**2+v_int**2)
        if wsp.any == 0.:
            raise Exception('Wind speed equals zero: division by zero')
        lon_int[:,N-n-1] = (lon_int[:,N-n] - (delta/np.cos(np.pi*lat_int[:,N-n]/180.)) * u_int/ wsp) % 360 # modulo 360; yields values between 0 and 360
        lat_int[:,N-n-1] = lat_int[:,N-n] - (delta * v_int/ wsp)
        wave_int[:,N-n-1] = waveRBS.ev(lat_int[:,N-n-1],lon_int[:,N-n-1])

    # set wave signal to zero outside of data domain
    # (The spline interpolation may actually extrapolate values and create
    # spurious data outside of the given data domain. To avoid contamination
    # of the envelope signal we set wave = 0 outside of the data domain
    #wave[np.where(lat_int > np.max(lat))] = 0.
    #wave[np.where(lat_int < np.min(lat))] = 0.
    # does not work at the moment ...
    #    if np.any(lat_int) > np.max(lat) or np.any(lat_int) < np.min(lat):
    #        print, np.where(lat_int > np.max(lat))
    #        print, np.where(lat_int < np.min(lat))
    #raise Exception('Contours integrated outside of data domain

    # apply Gauss filter
    # 'localization' of wave signal; see Zimin et al. 2006 for rationale:
    # "we choose 1/alpha to be roughly the length of the wave packet ...
    # as a fraction of the length of the latitude circle"
    gaussfilter = np.exp(-alpha**2 * (np.arange(2*N+1)-N)**2 / N**2)
    wave_filtered = wave_int * gaussfilter

    """
    calculate envelope and assign 'local' value
    """
    env = np.empty_like(LAT.flatten())
    for i in range(env.size):
        env[i] = hilbert(wave_filtered[i,:],2*N+1)[N] # value of envelope at the "midpoint" of the streamline

    # reshape array for 2D output
    envelope = np.reshape(env, [NX,NY])

    return envelope



def semigeostr_ct(data, PHI, PHImean, LAT, LON, kmin, kmax):
    """
    Semigeostrophic coordinate transformation as in Wolf and Wirth, 2015.
    Transformation is restricted to wavenumbers [kmin,kmax] of the geostrophic
    wind. Transformation should be applied also only to wavenumber-filtered
    data. This filter for data needs to be applied in the program calling this
    routine, though.

    DEPENDENCIES:
    numpy to be imported as np
    griddata imported from scipy.interpolate

    Expects PHI, PHImean, and data on regular/ EQUIDISTANT lat, lon grids.

    PHI: full geopotential
    PHImean: predefined background state.
         Transformation should use PHI-PHImean only (+ wavenumber filter).
    data: can contain several fields to be transformed. First two dimensions
         are expected to be lat, lon.
    This version assumes that all fields are periodic in longitude.
    Expects latitude values in the southern hemisphere to be negative:
    needs to be double-checked, though !!!
    LAT and LON are the respective fields of latitude and longitude
    """

    # check consistency of grid resolution
    if abs(LAT[0,0]-LAT[1,0]) != abs(LON[0,0]-LON[0,1]):
        raise IOError('There is a problem with the grid resolution')
    res = abs(LAT[0,0]-LAT[1,0])

    # get dimensions
    (ny,nx) = PHI.shape

    # necessary parameters
    a   = 6371000.                # earth radius in m
    Om  = 2*np.pi/(24.*3600.)        # angular velocity
    f   = 2*Om*np.sin(np.pi/180.*LAT)   # coriolis parameter

    # convert resolution to radians
    res = res * np.pi/180.

    #################################
    # Use wavenumber-filtered geostrophic wind (geopotential) anomaly only
    #################################
    PHIanom = PHI - PHImean # use anomaly of PHI
    PHI_WN  = zeros_like(PHI)
    # calculate (filtered) geostrophic wind
    for i_lat in range(np.shape(PHIanom[:,0])[0]):
        PHI_WN[i_lat,:] = wnedit(PHIanom[i_lat,:],kmin,kmax)
    #################################
    # extend domain to account for zonal periodicity
    PHIext = np.zeros((ny,nx+2))
    PHIext[:,1:nx+1] = PHI_WN
    PHIext[:,0] = PHIext[:,nx]
    PHIext[:,nx+1] = PHIext[:,0]

    # geostrophic wind from gradient of geopotential
    grady, gradx = np.gradient(PHIext, res)
    ug = -grady[:,1:nx+1]/(f*a)
    vg = gradx[:,1:nx+1]/(f*a*np.cos(np.pi/180.*LAT))
    #################################

    # calculate new coordinates
    LATn = LAT - (ug/f)*360/(2*a*pi)
    LONn = LON + (vg/f)*360/(2*a*pi*cos(LAT*pi/180.))

    # interpolate transformed data to original lat, lon grid
    #################################
    # In contrast to Wolf and Wirth (2015) we use the griddata routine,
    # which returns values within the convex envelope of the unstructured,
    # here i.e. transformed, grid (LATn, LONn).
    # To avoid loss of data, the domain is again periodically extended in
    # the zonal direction. Missing values should occur only close to the
    # northern and southern boundary and are filled with zeros.

    # Extend domain by the maximum of the longitudinal shift
    xtend = int(np.ceil(np.amax(abs(LONn-LON))))
##    print('max shift is', xtend)
    xLATn = np.zeros( (ny, nx + (2*xtend)) )
    xLONn = np.zeros( (ny, nx + (2*xtend)) )
    xdata = np.zeros( (ny, nx + (2*xtend)) )
    # copy eastern and western most coordinates, modulo 360 for longitude
    xLATn[:,0:xtend-1] = LATn[:,-xtend:-1]
    xLATn[:,xtend:-xtend] = LATn[:,:]
    xLATn[:,-xtend:-1] = LATn[:,-xtend:-1]
    xLONn[:,0:xtend-1] = LONn[:,-xtend:-1]-360.
    xLONn[:,xtend:-xtend] = LONn[:,:]
    xLONn[:,-xtend:-1] = LONn[:,-xtend:-1]+360.
    # copy associated data and prepare (reshape and flatten) arrays for griddata-routine
    gd_LAT = xLATn.reshape([xLATn.size,1]).flatten()
    gd_LON = xLONn.reshape([xLONn.size,1]).flatten()

    if len(np.shape(data)) > 2:
        for n in range(len(data)):
            xdata[:,0:xtend-1] = data[n][:,-xtend:-1]
            xdata[:,xtend:-xtend] = data[n][:,:]
            xdata[:,-xtend:-1] = data[n][:,-xtend:-1]
            gd_data = xdata.reshape([xdata.size,1]).flatten()
            ##        print(gd_data.size, gd_LON.size, gd_LAT.size)
            # finally, re-grid to original grid
            dummy = griddata((gd_LAT,gd_LON), gd_data, (LAT,LON), method='cubic', fill_value = 0.)
            data[n] = dummy.reshape([ny,nx])
    else:
        xdata[:,0:xtend-1] = data[:,-xtend:-1]
        xdata[:,xtend:-xtend] = data[:,:]
        xdata[:,-xtend:-1] = data[:,-xtend:-1]
        gd_data = xdata.reshape([xdata.size,1]).flatten()
        # finally, re-grid to original grid
        dummy = griddata((gd_LAT,gd_LON), gd_data, (LAT,LON), method='cubic', fill_value = 0.)
        data = dummy.reshape([ny,nx])

    return data


def wnedit(y, kmin, kmax):
    """
    Returns a one-dimensional function restricted to range of wavenumbers
    Modules used:     scipy for fft and ifft
    Input variables:  y                one-dimensional function
                      kmin, kmax       lower and upper limits of range of
                                       wavenumbers to be kept
    Return variable:  fedit
    """
    ffty = fft(y)
    mask = zeros(len(ffty))
    mask[kmin:kmax+1] = 1  # values outside the selected range remain zero
    mafft = ffty*mask
    fedit = ifft(mafft)
    fedit = 2*fedit.real # Since the ignored negative frequencies would contribute the same as the positive ones
    if kmin == 0:
        fedit = fedit - ffty.real[0]/len(ffty)
    elif kmin > 0:
        fedit = fedit + ffty.real[0]/len(ffty)
    return fedit


def wnedit2Dlon(y,kmin, kmax):
    """
    Takes a fields and returns only range of wavenumbers (in longitude!)
    """
    (ny, nx) = y.shape
    res = np.empty([ny, nx])
    for i in range(ny):
        res[i,:] =  wnedit(y[i,:], kmin, kmax)

    return res


def hilbert(y, N):
    """
    Returns the envelope of a one-dimensional function using the method
    by Marple (1999), also used by Zimin (2003).
    Modules used:     scipy for fft and ifft
    Input variables:  y                one-dimensional function
                      N                number of data points of y
    Return variable:  z                envelope function
    """
    z = fft(y)
    z[(int(N/2)+1):N] = 0
    z = 2*z
    z[0] = z[0]/2
    z[int(N/2)] = z[int(N/2)]/2
    z = ifft(z)
    z = abs(z)
    return z


def globe():
    map = Basemap(projection='ortho',lat_0=50,lon_0=0,resolution='l')
    map.drawcoastlines(linewidth=0.25)
    map.drawcountries(linewidth=0.25)
    map.fillcontinents(color='coral',lake_color='aqua')
    map.drawmapboundary(fill_color='aqua')
    map.drawmeridians(np.arange(0,360,30))
    map.drawparallels(np.arange(-90,90,30))
    return map


def worldmap(corners):
    """
    Draws costlines, meridians and longitudes as lat-long-plot
    Modules used:     mpl_toolkits.basemap
    Input variables:  llo, lla, ulo, ula
                                      lat-lon coordinates of lower-left
                                      and upper-right corner of plot
                                      llo lower-left longitude
                                      lla lower-laft latitude
                                      ulo upper-right longitude
                                      ula upper-right latitude
    Return variable   none, jenvdust a plotted worldmap
    """
    [llo, lla, ulo, ula] = corners
    map = Basemap(llo, lla, ulo, ula, rsphere=(6378137.00, 6356752.3142), resolution='l', projection='cyl',
                  area_thresh=10000, lat_0=50., lon_0=0., lat_ts=20.)
    map.drawcoastlines(linewidth=0.1)
    map.drawparallels(np.arange(20., 100., 20.), labels=[True, False, False, False], linewidth=0.25, fontsize=12)
    map.drawmeridians(np.arange(llo, ulo, 60.), labels=[False, False, False, True], linewidth=0.25, fontsize=12)
    return map


def shiftdata(field, lats, lons):
    """
    Rearrangement of lat-lon array to have zero meridian centered for plotting
    Input variables:    field          two-dimenisional field on lat-lon grid
                        lats,lons      latitudes and longitudes of data points
                                       of field
    Return variable:    field          shifted field centered around 180deg longitude
    """

    lonh = int(lons/2)
    temp = np.zeros([lats, lons])
    temp[:, 0:lonh] = field[:, lonh:lons]
    temp[:, lonh:lons] = field[:, 0:lonh]
    field = temp
    temp[0:lats, :] = field[np.arange(int(lats-1), -1, -1), :]
    field = temp
    return field
    temp = None


def shiftdata1(field, lats, lons):
    """
    Rearrangement of lat-lon array to have zero meridian centered for plotting
    Input variables:    field          two-dimenisional field on lat-lon grid
                        lats,lons      latitudes and longitudes of data points
                                       of field
    Return variable:    field          shifted field centered around 180deg longitude
    """

    lonh = int(lons/2)
    temp = np.zeros([lons,lats])
    temp[0:lonh,:] = field[lonh:lons,:]
    temp[lonh:lons,:] = field[0:lonh,:]
    field = temp
##    temp[:,0:lats] = field[:,np.arange(int(lats-1), -1, -1)]
##    field = temp
    return field
    temp = None


def z_field(field, lats, lons):
    """
    Cyclic ompletion of two-dimensional array
    Input variables:    field          two-dimenisional field on lat-lon grid
                        lats,lons      latitudes and longitudes of data points
                                       of field
    Return variable:    zfield =       field with repetition of left-most
                                       column at right-most boundary
    """
    zfield = np.zeros([lats, lons])
    zfield[:, 0:lons-1] = field
    zfield[:, lons-1] = zfield[:, 0]
    zfield = zfield.transpose()
    return zfield


def double_threshold(field, sizelon, gridres, lats, lat_range=(10, 70)):
    """
    Computation of double threshold for envelopss following Wolf (2015)
    Modules used:       numpy
    Input variables:    field          two-dimenisional envelope field on
                                       lat-lon grid
                        sizelon        number of data points of field in lon
                                       direction
                        gridres        lat-lon resolution of field
                        lats           vertor of latitudes
    Constants:          tauowp, tauuwp, deltatau, taustern
                                       Constants given by Wolf (2015),
                                       see pp ???.
    Return variables:   taumin,taumax  lower and uper threshold value
    """
    envsum = 0.
    latsum = 0.
    tauowp = 30.
    tauuwp = 25.
    deltatau = 8.
    taustern = 2.9
    pihalbe = np.pi/2
    lat_ind = np.where((lats >= lat_range[0]) & (lats <= lat_range[1]))[0]
    for ilon in np.arange(sizelon):
        for jlat in lat_ind:
            envsum += field[ilon, jlat]*np.cos(np.pi*lats[jlat]/180)
            latsum += np.cos(np.pi*lats[jlat]/180)
    envmean = envsum/latsum
    taumin = (tauuwp
              + deltatau*np.arctan((taustern*envmean-tauuwp)
                                   / (deltatau*0.8))/pihalbe)
    taumax = (tauowp
              + deltatau*np.arctan((taustern*envmean-tauowp)
                                   / (deltatau*0.6))/pihalbe)
    return float(taumin), float(taumax)


def rossby_wave_packets_diag(u, v, z, lon=None, lat=None, date=None, lat_range=(10, 90), semigeostr=False, zimim2006=False):
    """
    This function applies the Rossby wave packet diagnostics developed at the Institute for Atmospheric Physics
    (Johannes Gutenberg-University, Mainz). Several parts of it are based on Gabriel Wolf's comprehensive set
    of matlab tools, which can't be appreciated high enough.

    This function is a wrapper around individual components of the analysis and is intended for convenient usage.

    Parameters
    ----------
    u : xarray.DataArray
            u-wind component in 300 hPa (time, lev, lat, lon) [m s-1]

    v : xarray.DataArray
            v-wind component in 300 hPa (time, lev, lat, lon) [m s-1]

    z : xarray.DataArray (time, lev, lat, lon) [m^2 s-2]
            geopotential in 300 hPa

    lon : xarray.DataArray or np.ndarray
            longitude coordinate of the input arrays

    lat : xarray.DataArray or np.ndarray
            latitude coordinate of the input arrays

    date : datetime
            if provided, the calculation will be done for this specific date.

    lat_range : tuple
            (lat_min, lat_max) the latitude range to apply the diagnostic on

    semigeostr : bool
            if True, semigeostrophic transformation is applied to the input data following Wolf and Wirth (2015) [3]_.

    zimim2006 : bool
            if True, the calculation follows Zimim (2006) [2]_.

    Returns
    -------
    xarray.Dataset
            The returned dataset contains all information necessary for plotting, it can easily be stored into a netcdf
            file for later processing.

    References
    ----------
    .. [1]  Zimin, A.V., I. Szunyogh, D.J. Patil, B.R. Hunt, and E. Ott, 2003: Extracting Envelopes of Rossby Wave
            Packets. Mon. Wea. Rev., 131, 1011-1017, https://doi.org/10.1175/1520-0493(2003)131<1011:EEORWP>2.0.CO;2

    .. [2]  Zimin, A.V., I. Szunyogh, B.R. Hunt, and E. Ott, 2006: Extracting Envelopes of Nonzonally Propagating
            Rossby Wave Packets. Mon. Wea. Rev., 134, 1329-1333, https://doi.org/10.1175/MWR3122.1

    .. [3]  Wolf, G. and V. Wirth, 2015: Implications of the Semigeostrophic Nature of Rossby Waves for Rossby Wave
            Packet Detection. Mon. Wea. Rev., 143, 26-38, https://doi.org/10.1175/MWR-D-14-00120.1
    """

    # get the coordinates
    lon, lat = get_coordinates_from_xarray(u, lon, lat, create_mesh=False, only_spatial_dims=False)

    # remove the level dimension if present
    if u.ndim == 4:
        if u.shape[1] == 1:
            u = u[:, 0, ...]
    if v.ndim == 4:
        if v.shape[1] == 1:
            v = v[:, 0, ...]
    if z.ndim == 4:
        if z.shape[1] == 1:
            z = z[:, 0, ...]

    # select the requested latitude range
    lat_ind = np.where((lat >= lat_range[0]) & (lat <= lat_range[1]))[0]
    lat = np.asarray(lat[lat_ind])
    if lat[0] > lat[1]:
        lat_ind = np.flip(lat_ind, 0)
        lat = np.asarray(lat[lat_ind])
    lon = np.asarray(lon)
    u = u[:, lat_ind, :]
    v = v[:, lat_ind, :]
    z = z[:, lat_ind, :]

    # select the requested time step
    u_t = np.asarray(u.sel(time=date))
    v_t = np.asarray(v.sel(time=date))
    z_t = np.asarray(z.sel(time=date))

    # select the time steps for the 21-day average
    z_4mean = z.sel(time=slice(date - timedelta(days=10, hours=11), date + timedelta(days=10, hours=11)))

    # is the number of time steps large enough for the calculation of mean values?
    enough_values_for_mean = z_4mean.shape[0] >= 21

    # calculate 21-day mean of geopotential
    if enough_values_for_mean:
        z_mean = np.asarray(z_4mean.mean(dim="time"))

    # for zimim2006, we also need mean wind values
    if zimim2006:
        if not enough_values_for_mean:
            raise ValueError("for the Zimim 2006 variant of the calculation, at least values for 21 days are required!")

        # select the time steps for the 21-day average
        u_4mean = u.sel(time=slice(date - timedelta(days=10, hours=11), date + timedelta(days=10, hours=11)))
        v_4mean = v.sel(time=slice(date - timedelta(days=10, hours=11), date + timedelta(days=10, hours=11)))

        # calculate 21-day mean of geopotential
        u_mean = np.asarray(u_4mean.mean(dim="time"))
        v_mean = np.asarray(v_4mean.mean(dim="time"))

        # for Zimim 2006, we also need the non-semi-geostrophic values
        z_t2 = z_t
        u_t2 = u_t
        v_t2 = v_t

    # some constants
    kmin = 4
    kmax = 17

    # apply semi-geostrophic transformation
    if semigeostr or zimim2006:
        if not enough_values_for_mean:
            raise ValueError("for the Semi-geostrophic variant of the calculation, at least values for 21 days are required!")

        lon_mesh, lat_mesh = np.meshgrid(lon, lat)
        data = [u_t, v_t, z_t]
        data_sg = semigeostr_ct(data, z_t, z_mean, lat_mesh, lon_mesh, kmin, kmax)
        u_t = data_sg[0]
        v_t = data_sg[1]
        z_t = data_sg[2]

    # perform the actual calculation
    lon_cycl = np.append(lon, lon[0] + 360)
    ZU = z_field(u_t, lat.size, lon.size + 1)
    ZV = z_field(v_t, lat.size, lon.size + 1)
    ZUV = np.hypot(ZU, ZV)
    ZPHI = z_field(z_t / 9.81, lat.size, lon.size + 1)
    if enough_values_for_mean:
        ZPHI_mean = z_field(z_mean / 9.81, lat.size, lon.size + 1)

    # calculation according to Zimim 2006
    if zimim2006:
        kmin_env = 4
        kmax_env = 14
        kmin_sq = 4
        kmax_sq = 14
        wsp_perd = wsp_perpendicular(u_t, v_t, u_mean, v_mean)
        wsp_perd_wnf = wnedit2Dlon(wsp_perd, kmin_sq, kmax_sq)
        data = [wsp_perd_wnf]
        result = semigeostr_ct(data, z_t2, z_mean, lat_mesh, lon_mesh, kmin_env, kmax_env)
        wsp_sg = result[0]
        # we need lon from 0 to 360?
        if np.any(lon > 0):
            lon_0_360 = lon + lon.min() * -1
        else:
            lon_0_360 = lon
        da = envelope_along_streamline(u_mean, v_mean, wsp_sg, lat, lon_0_360, 1., 4.)
        ZWSP = z_field(wsp_sg, lat.size, lon.size + 1)
        ZENV = z_field(da, lat.size, lon.size + 1)
    else:
        ZENV = np.zeros([lat.size, lon.size + 1])
        ZVF = np.zeros([lat.size, lon.size + 1])
        for ilat in range(1, lat.size):
            ZVF[ilat, :] = wnedit(ZV[..., ilat], kmin, kmax)
            ZENV[ilat, :] = hilbert(ZVF[ilat, :], lon.size)
        ZENV = ZENV.transpose()

    # compute double threshold
    gridres = float(abs(lon[1]-lon[0]))
    tauun, tauob = double_threshold(ZENV, lon.size, gridres, lat)

    # assign points with ZENV > tauob to RWP objects
    ZOBJ = ZENV/tauob
    ZOBJ = np.ma.MaskedArray(ZOBJ, ZOBJ < 1)
    ZOBJ = ZOBJ.astype(int)+1.e-15

    # create result dataset for return
    result = xarray.Dataset({"lon": ("lon", lon_cycl),
                             "lat": ("lat", lat),
                             "ZU": (["lon", "lat"], ZU),
                             "ZV": (["lon", "lat"], ZV),
                             "ZUV": (["lon", "lat"], ZUV),
                             "ZPHI": (["lon", "lat"], ZPHI),
                             "ZENV": (["lon", "lat"], ZENV),
                             "ZOBJ": (["lon", "lat"], ZOBJ)})

    # only add mean values if calculated
    if enough_values_for_mean:
        result["ZPHI_mean"] = (["lon", "lat"], ZPHI_mean)
        result["ZPHI_mean"].attrs["title"] = "Height of 300 hPa surface, 21 days average"
        result["ZPHI_mean"].attrs["units"] = "gpm"

    # add description for the variables
    result["ZU"].attrs["title"] = "Zonal wind speed"
    result["ZU"].attrs["units"] = "m s-1"
    result["ZV"].attrs["title"] = "Meridional wind speed"
    result["ZV"].attrs["units"] = "m s-1"
    result["ZUV"].attrs["title"] = "Horizontal wind speed"
    result["ZUV"].attrs["units"] = "m s-1"
    result["ZENV"].attrs["title"] = "Envelope of meridional wind"
    result["ZENV"].attrs["units"] = "m s-1"
    result["ZPHI"].attrs["title"] = "Height of 300 hPa surface"
    result["ZPHI"].attrs["units"] = "gpm"
    result["ZOBJ"].attrs["title"] = "RWP objects"
    result["ZOBJ"].attrs["levelu"] = tauun
    result["ZOBJ"].attrs["levelo"] = tauob

    # modify title for semi-geostrophic data
    if semigeostr or zimim2006:
        for varname in ["ZU", "ZV", "ZUV", "ZENV", "ZPHI", "ZOBJ"]:
            result[varname].attrs["title"] += ", semi-geostrophic"

    # add zimim2006 variable
    if zimim2006:
        result["ZWSP"] = (["lon", "lat"], ZWSP)
        result["ZWSP"].attrs["title"] = "Wave signal perp. to background flow, semi-geostrophic"
        result["ZWSP"].attrs["units"] = "m s-1"
        for varname in ["ZENV", "ZOBJ"]:
            result[varname].attrs["title"] += ", Z2006"

    # rotate the result back to the original shape and add a time coordinate
    result.coords["time"] = date
    result = result.transpose().expand_dims("time", 0)
    return result


def __plot(title, Z, levz, colormap, colorbar, levu, levo, filled=True, fig=None, subplot_args=None):
    """
    plot filled contours
    """
    # remove time coordinate if present
    if Z.ndim == 3 and Z.dims[0] == "time":
        if Z.shape[0] > 1:
            raise ValueError("plot only one time-step at a time!")
        Z = Z.squeeze("time")

    # create the contour plot
    fig, ax = contour(Z, levels=levz,
                      cmap=colormap,
                      colorbar=colorbar,
                      filled=filled,
                      figure=fig,
                      coastlines_kwargs={"color": (0., 0., 0., 0.5)},
                      subplot_args=subplot_args)
    ax.set_title(title)

    # add overlay
    if levu != 0:
        fig, ax = contour(Z, figure=fig, axes=ax, levels=[levu],
                          colorbar=False, filled=False, coastlines=False,
                          linewidths=1.0, linestyles='dashed', dashes=[(0, (3.0, 2.0))], colors="k", subplot_args=subplot_args)
        fig, ax = contour(Z, figure=fig, axes=ax, levels=[levo],
                          colorbar=False, filled=False, coastlines=False,
                          linewidths=2.0, linestyles='-', colors="k", subplot_args=subplot_args)
    return fig, ax


def rossby_wave_packets_plot(result, plot_numbers=None):
    """
    Create standard plots for rossby wave packet diagnostic

    Parameters
    ----------
    plot_numbers : int or list of int
            number of a plot or list of plots to create.

    result : xarray.Dataset
            output of rossby_wave_packets_diag

    Examples
    --------
    Read a file, which contains u, v, geopotential on a regular grid. An example is available for download here:
    https://syncandshare.lrz.de/dl/fiTi9jXWQAuk28ugV4GpZbEB/example_post_rwp_01.nc

    >>> nc = read("example_post_rwp_01.nc"))                                                    # doctest: +SKIP

    calculate the diagnostic for one point in time

    >>> rwp = rossby_wave_packets_diag(nc["u"], nc["v"], nc["z"], date=datetime(2002, 8, 7))    # doctest: +SKIP

    create standard plots

    >>> fig, ax = rossby_wave_packets_plot(rwp)                                                 # doctest: +SKIP

    .. figure:: images/example_post_rwp_01.png


    """

    # levels for individual plots
    lev_phi = [8700, 8800, 8900, 9000, 9100, 9200, 9300, 9400, 9500, 9600, 9700]
    lev_vabs = [10, 20, 30, 40, 50]
    lev_v = [-60, -45, -30, -15, 15, 30, 45, 60]
    lev_env = [10, 20, 30, 40, 50]
    lev_obj = [0, 1, 1000]

    # colorbar for objects
    colobj = mpl.colors.ListedColormap((np.array([[255, 255, 255], [255, 0, 0]]) / 255.))

    # how many plots should be created?
    if plot_numbers is None:
        plot_numbers = [1, 2, 3, 4, 5, 6]
    if type(plot_numbers) == int:
        nplots = 1
        plot_numbers = [plot_numbers]
    else:
        nplots = len(plot_numbers)

    # arrangement of sub-plots
    nsubplots = 0
    fig, ax = None, []
    if 1 in plot_numbers and "ZPHI_mean" in result:
        nsubplots += 1
        fig, _ax = __plot("%s (%s)" % (result["ZPHI_mean"].attrs["title"], result["ZPHI_mean"].attrs["units"]), result["ZPHI_mean"], lev_phi, 'RdYlBu_r', True, 0, 0, fig=fig, subplot_args=(nplots, 1, nsubplots))
        ax.append(_ax)
    if 2 in plot_numbers:
        nsubplots += 1
        fig, _ax = __plot("%s (%s)" % (result["ZPHI"].attrs["title"], result["ZPHI"].attrs["units"]), result["ZPHI"], lev_phi, 'RdYlBu_r', True, 0, 0, fig=fig, subplot_args=(nplots, 1, nsubplots))
        ax.append(_ax)
    if 3 in plot_numbers:
        nsubplots += 1
        fig, _ax = __plot("%s (%s)" % (result["ZUV"].attrs["title"], result["ZUV"].attrs["units"]), result["ZUV"], lev_vabs, 'YlGnBu', True, 0, 0, fig=fig, subplot_args=(nplots, 1, nsubplots))
        ax.append(_ax)
    if 4 in plot_numbers:
        nsubplots += 1
        fig, _ax = __plot("%s (%s)" % (result["ZV"].attrs["title"], result["ZV"].attrs["units"]), result["ZV"], lev_v, 'bwr', True, 0, 0, fig=fig, subplot_args=(nplots, 1, nsubplots))
        ax.append(_ax)
    if 5 in plot_numbers:
        nsubplots += 1
        fig, _ax = __plot("%s (%s)" % (result["ZENV"].attrs["title"], result["ZENV"].attrs["units"]), result["ZENV"], lev_env, 'Greens', True, result["ZOBJ"].attrs["levelu"], result["ZOBJ"].attrs["levelo"], fig=fig, subplot_args=(nplots, 1, nsubplots))
        ax.append(_ax)
    if 6 in plot_numbers:
        nsubplots += 1
        ZOBJ = result["ZOBJ"].copy().fillna(0.0)
        fig, _ax = __plot(ZOBJ.attrs["title"], ZOBJ, lev_obj, colobj, "empty", 0, 0, fig=fig, subplot_args=(nplots, 1, nsubplots))
        ax.append(_ax)

    return fig, ax
