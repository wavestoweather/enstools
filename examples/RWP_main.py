"""
This script computes and plots envelopes of meridional wind
as well as RWP objects.

Authors:   Joachim Eichhorn, Georgios Fragkoulidis,
           Ilona Glatt, Gabriel Wolf, Gil Fleger

The script has been tested in a Python 3.5 environment using the
anaconda framework. Required modules are listed below.

"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
#import subprocess
import datetime as dt
from datetime import timedelta
from RWP_tools import *  # RWPtools.py must be in same folder as this script
from netCDF4 import Dataset
from pylab import *


###############################################################################
# USER INPUT
###############################################################################
#    
# specify fields to be plotted, set plotall = 1 to plot all
#
plotall = 1
plot00 = 0  # 21 days average of geopotential (gpdam)
plot01 = 0  # Geopotential (gpdam)
plot02 = 0  # Horizontal wind speed (m/s)
plot03 = 0  # Meridional wind speed (m/s)
plot04 = 0  # Envelope of meridional wind (m/s)
plot05 = 0  # RWP objects
plot06 = 0  # Geopotential, semi-geostrophic (gpdam)
plot07 = 0  # Horizontal wind speed, semi-geostrophic (m/s)
plot08 = 0  # Meridional wind speed, semi-geostrophic (m/s)
plot09 = 0  # Envelope of meridional wind, semi-geostrophic (m/s)
plot10 = 0  # RWP objects, semi-geostrophic
plot11 = 0  # Envelope of wave signal, semi-geostrophic (m/s), Z2006
plot12 = 0  # RWP objects, semi-geostrophic, Z2006
plot13 = 0  # Wave signal perp. to background flow, semi-geostrophic (m/s)
#
# specify date between 20020611 and 20061004
#
yearstart = 2002; monstart  = 8; daystart  = 7; hourstart = 0
#
# specify plot domain
# [lower-left lon, lower-left lat, upper-right lon, upper-right lat]
plotcorners = [-180, 15, 180, 80]
#
###############################################################################


#########################
## plot fields and titles
#########################
plotnrs = ['phimean', 'phi', 'vabs', 'v300', 'env', 'obj', 'phi_sg',
           'vabs_sg', 'v300_sg', 'env_sg', 'obj_sg', 'env_2006', 'obj_2006',
           'wave_sig_sg']

## contour levels
lev_v = [-60, -45, -30, -15, 15, 30, 45, 60]
lev_difv = [-40, -30, -20, -10, 10, 20, 30, 40]
lev_env = [10, 20, 30, 40, 50]
lev_dife = [-20, -15, -10, -5, 5, 10, 15, 20]
lev_vabs = [10, 20, 30, 40, 50]
#lev_phi = [7700,7800,7900,8000,8100,8200,8300,8400,8500,8600,8700,8800,8900,9000,9100,9200,9300,9400,9500,9600,9700]
lev_phi = [8600, 8700, 8800, 8900, 9000, 9100, 9200, 9300, 9400, 9500, 9600]
lev_obj = [0, 1, 1000]

## colorbar for objects
colobj = mpl.colors.ListedColormap((np.array([[255, 255, 255], [255, 0, 0]])/255.))

## paths
indir = "Data/"
plotdir = "Plots/"

## the file sample.* contains u, v and phi on 300 hPa
## from 20020601 to 20021014, 0 and 12 UTC, 2x2 grid
## all ERAinterim data used courtesy of ECMWF, Reading, UK
#infile    = indir + "sample.grib"
ncfile    = indir + "sample.nc"
#subprocess.Popen(['cdo','-f','nc','copy',infile,ncfile]).wait()

ncinfil = Dataset(ncfile, 'r', format='NETCDF4')
lats = ncinfil.variables['lat'][0:45]
lons = ncinfil.variables['lon'][:]

## read data, reduced to  northern hemisphere
phi = ncinfil.variables['z'][:, 0, 0:45, :]
u = ncinfil.variables['u'][:, 0, 0:45, :]
v = ncinfil.variables['v'][:, 0, 0:45, :]
lats = ncinfil.variables['lat'][0:45]
lons = ncinfil.variables['lon'][:]
## extract grid resolution from nc-file
lonsize   = np.shape(lons)[0]
latsize   = np.shape(lats)[0]
gridres   = int(360/lonsize)

time = ncinfil.variables['time']
reftime = dt.datetime(2002, 6, 1, 0)
starttime = dt.datetime(yearstart, monstart, daystart, hourstart)
starthours = (starttime - reftime)// timedelta(seconds=3600)
itmax = time.size

for itime in np.arange(itmax):
    if time[itime] == starthours:
        break

## data for date chosen
phii = phi[itime]
vv = v[itime]
uu = u[itime]

## compute simple 21day-mean of phi, u, v
meanstart = itime-20 # 20 <--> 2 fields per day
meanstop = itime+20
phimean = np.mean(phi[meanstart:meanstop+1,:,:], axis=0)
umean = np.mean(u[meanstart:meanstop+1,:,:], axis=0)
vmean = np.mean(v[meanstart:meanstop+1,:,:], axis=0)


## specify plot coordinates
lati = np.arange(2, 90+gridres, gridres)
longi = np.arange(-180, 180+gridres, gridres)
Y, X = np.meshgrid(lati, longi)  # longitudes are on the x-axis

# some constants
kmin = 4
kmax = 17

## prepare fields for plotting, center around 0 longitude
## all plot fields' names start with a Z
v300 = shiftdata(vv, latsize, lonsize)
u300 = shiftdata(uu, latsize, lonsize)
phi300 = shiftdata(phii, latsize, lonsize)

ZV = z_field(v300, latsize, lonsize+1)
ZU = z_field(u300, latsize, lonsize+1)
ZUV = np.hypot(ZU,ZV)
ZPHI = z_field(phi300/10., latsize, lonsize+1)
ZPHImean = shiftdata(phimean, latsize, lonsize)
ZPHImean = z_field(ZPHImean/10., latsize, lonsize+1)
ZENV = np.zeros([latsize, lonsize+1])
ZVF = np.zeros([latsize, lonsize+1])
for ilat in range(1, int(latsize)):
    ZVF[ilat, :] = wnedit(ZV[:, ilat], kmin, kmax)
    ZENV[ilat, :] = hilbert(ZVF[ilat, :], lonsize)
ZENV = ZENV.transpose()

## compute double threshold
[tauun, tauob] = double_threshold(ZENV, lonsize, gridres, lati)
levelu = [tauun]
levelo = [tauob]

## assign points with ZENV > tauob to RWP objects
ZOBJ = ZENV/tauob
ZOBJ = np.ma.MaskedArray(ZOBJ, ZOBJ < 1)
ZOBJ = ZOBJ.astype(int)+1.e-15

if plotall == 1 or plot00 == 1:
    plotitf(plotnrs[0], "Height of 300 hPa surface, 21 days average (gpdam)", plotcorners, X, Y, ZPHImean, lev_phi, 'RdYlBu_r', 1, 0, 0)
if plotall == 1 or plot01 == 1:
    plotitf(plotnrs[1], "Height of 300 hPa surface (gpdam)", plotcorners, X, Y, ZPHI, lev_phi, 'RdYlBu_r', 1, 0, 0)
if plotall == 1 or plot02 == 1:
    plotitf(plotnrs[2], "Horizontal wind speed (m/s)", plotcorners, X, Y, ZUV, lev_vabs, 'YlGnBu', 1, 0, 0)
if plotall == 1 or plot03 == 1:
    plotitf(plotnrs[3], "Meridional wind speed (m/s)", plotcorners, X, Y, ZV, lev_v, 'bwr', 1, 0, 0)
if plotall == 1 or plot04 == 1:
    plotitf(plotnrs[4], "Envelope of meridional wind (m/s)", plotcorners, X, Y, ZENV, lev_env, 'Greens', 1, levelu, levelo)
if plotall == 1 or plot05 == 1:
    plotitf(plotnrs[5], "RWP objects", plotcorners, X, Y, ZOBJ, lev_obj, colobj, 0, 0, 0)

###########################################################
## same again, but now apply semigeostrophic transformation
###########################################################

## needs original grid
[lo, la] = np.meshgrid(lons, lats)
data = np.zeros([latsize, lonsize, 3])    # 3 is hardcoded here...
data = [uu, vv, phii]
result = np.zeros(np.shape(data))
result = semigeostr_ct(data, phii, phimean, la, lo, kmin, kmax)
usg = result[0]
vsg = result[1]
phisg = result[2]
v300sg = shiftdata(vsg, latsize, lonsize)
u300sg = shiftdata(usg, latsize, lonsize)
phi300sg = shiftdata(phisg, latsize, lonsize)

ZVsg = z_field(v300sg, latsize,  lonsize+1)
ZUsg = z_field(u300sg, latsize,  lonsize+1)
ZUVsg = np.hypot(ZUsg,ZVsg)
ZPHIsg = z_field(phi300sg/10., latsize, lonsize+1)
ZENVsg = np.zeros([latsize, lonsize+1])
ZVFsg = np.zeros([latsize, lonsize+1])
for ilat in range(1, int(latsize)):
    ZVFsg[ilat, :] = wnedit(ZVsg[:, ilat], kmin, kmax)
    ZENVsg[ilat, :] = hilbert(ZVFsg[ilat, :], lonsize)
ZENVsg = ZENVsg.transpose()

[tauun, tauob] = double_threshold(ZENVsg, lonsize, gridres, lati)
levelu = [tauun]
levelo = [tauob]
ZOBJsg = ZENVsg/tauob
ZOBJsg = np.ma.MaskedArray(ZOBJsg, ZOBJsg < 1)
ZOBJsg = ZOBJsg.astype(int)+1.e-15

if plotall == 1 or plot06 == 1:
    plotit(plotnrs[6], "Geopotential, semi-geostrophic (gpdam)", plotcorners, X, Y, ZPHIsg, lev_phi)
if plotall == 1 or plot07 == 1:
    plotitf(plotnrs[7], "Horizontal wind speed, semi-geostrophic (m/s)", plotcorners, X, Y, ZUVsg, lev_vabs, 'YlGnBu', 1, 0, 0)
if plotall == 1 or plot08 == 1:
    plotitf(plotnrs[8], "Meridional wind speed, semi-geostrophic (m/s)", plotcorners, X, Y, ZVsg, lev_v, 'bwr', 1, 0, 0)
if plotall == 1 or plot09 == 1:
    plotitf(plotnrs[9], "Envelope of meridional wind, semi-geostrophic (m/s)", plotcorners, X, Y, ZENVsg, lev_env, 'Greens', 1, levelu, levelo)
if plotall == 1 or plot10 == 1:
    plotitf(plotnrs[10], "RWP objects, semi-geostrophic", plotcorners, X, Y, ZOBJsg, lev_obj, colobj, 0, 0, 0)

######################
## Zimin et al. (2006)
######################
kmin_env = 4
kmax_env = 14
kmin_sq = 4
kmax_sq = 14
wsp_perd = wsp_perpendicular(u300sg,v300sg,umean,vmean)
wsp_perd_wnf = wnedit2Dlon(wsp_perd, kmin_sq, kmax_sq)
data = [wsp_perd_wnf]
result = semigeostr_ct(data,phii,phimean, la, lo, kmin_env, kmax_env)
wsp_sg = result[0]
da = envelope_along_streamline(umean, vmean, wsp_sg, -lats, lons, 1., 4.)

##da = shiftdata(da, latsize, lonsize)
##da = shiftdata(da, latsize, lonsize)

ZWSP = z_field(wsp_sg, latsize, lonsize+1)
ZENV2006 = z_field(da, latsize,  lonsize+1)
[tauun, tauob] = double_threshold(ZENV2006, lonsize, gridres, lati)
levelu = [tauun]
levelo = [tauob]
ZOBJ2006 = ZENV2006/tauob
ZOBJ2006 = np.ma.MaskedArray(ZOBJ2006, ZOBJ2006 < 1)
ZOBJ2006 = ZOBJ2006.astype(int)+1.e-15

if plotall == 1 or plot11 == 1:
    plotitf(plotnrs[11], "Envelope of wave signal, semi-geostrophic (m/s), Z2006", plotcorners, X, Y, ZENV2006, lev_env, 'Greens', 1, levelu, levelo)
if plotall == 1 or plot12 == 1:
    plotitf(plotnrs[12], "RWP objects, semi-geostrophic, Z2006", plotcorners, X, Y, ZOBJ2006, lev_obj, colobj, 0, 0, 0)
if plotall == 1 or plot13 == 1:
    plotitf(plotnrs[13], "Wave signal perp. to background flow, semi-geostrophic (m/s)", plotcorners, X, Y, ZWSP, lev_v, 'bwr', 1, 0, 0)
