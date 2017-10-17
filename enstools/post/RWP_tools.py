"""
This module contains python scripts to perform several RWP diagnostics.

Authors:   Joachim Eichhorn, Georgios Fragkoulidis, Michael Riemer,
           Ilona Glatt, Gabriel Wolf, Gil Fleger

The scripts have been tested in a Python 3.5 environment using the
anaconda framework. Required modules are listed below.

"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import netCDF4 as nc
from netCDF4 import Dataset, netcdftime
from pylab import *
from mpl_toolkits.basemap import Basemap
from scipy import fft, ifft, arange
from scipy.fftpack import fftfreq, fftshift
from scipy.interpolate import griddata, RectBivariateSpline


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


def double_threshold(field, sizelon, gridres, lats):
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
    for ilon in np.arange(sizelon):
        for jlat in np.arange(int(10/gridres), int(70/gridres)):
            envsum += field[ilon, jlat]*np.cos(np.pi*lats[jlat]/180)
            latsum += np.cos(np.pi*lats[jlat]/180)
    envmean = envsum/latsum
    taumin = (tauuwp
              + deltatau*np.arctan((taustern*envmean-tauuwp)
                                   / (deltatau*0.8))/pihalbe)
    taumax = (tauowp
              + deltatau*np.arctan((taustern*envmean-tauowp)
                                   / (deltatau*0.6))/pihalbe)
    return taumin, taumax


def plotitf(plotnr, title, corners, X, Y, Z, levz, colormap, ibar, levu, levo):
    """
    plot filled contours
    """
    fig = plt.figure(facecolor='1.', figsize=[11, 4])
    plt.title(title)
    map = worldmap(corners)
    CP = map.contourf(X, Y, Z, levz, cmap=colormap, extend='both', latlon='true')
    if ibar == 1:
        cbar = map.colorbar(CP, location='bottom', pad="18%", size="5%")
        cbar.ax.tick_params(labelsize=10)

    if (levu != 0):
        CPu = map.contour(X, Y, Z, levu, colors='k',
                          linewidth=0.5, linestyles='dashed', latlon='true')
        CPo = map.contour(X, Y, Z, levo, colors='k',
                          linewidth=0.5, latlon='true')
        for c in CPu.collections:

            c.set_dashes([(0, (3.0, 2.0))])

    savefig('Plots/' + plotnr + '.png', bbox_inches='tight')


def plotit(plotnr, title, corners, X, Y, Z, levz):
    """
    plot contours
    """
    fig = plt.figure(facecolor='1.', figsize=[11, 4])
    plt.title(title)
    map = worldmap(corners)
    CP = map.contour(X, Y, Z, levz, latlon='true')
    savefig('Plots/' + plotnr + '.png', bbox_inches='tight')
