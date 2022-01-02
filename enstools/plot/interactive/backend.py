from enum import Enum, auto
from abc import ABCMeta, abstractmethod

from enstools.misc import generate_coordinates, spherical2cartesian
from enstools.interpolation import nearest_neighbour

import numpy as np
from scipy.spatial import cKDTree
from math import sqrt, floor
from collections import OrderedDict

import cartopy
import matplotlib.pyplot as plt



class Backend(Enum):
    """
    Enum class that defining a library for plotting.
    """
    PLOTLY = auto()
    BOKEH  = auto()
    
class Stream(Enum):
    """
    Enum class that defining a type of streamlines.
    """
    LINE   = auto()
    VECTOR = auto()
    
class Function(Enum):
    """
    Enum class that defining a Plotly function used for a plotting.
    """
    TRISURF = auto()
    MESH3D  = auto()
    

class InteractiveBackend(metaclass=ABCMeta):
    """
    Abstract backend class that will be implemented for different plotting backends. Functions that are used for all
    backends can be implemented directly here.
    """
        
    @classmethod
    def icosahedral_to_regular_coords(cls, data_icosahedral, clon, clat, cell_area=None):
        """
        Converts icosahedral coordinates to regular-lat-lon coordinates.

        Parameters
        ----------
        data_icosahedral : xarray
            Values of a variable in icosahedral coordinates.

        clon: xarray
            Longitude coordinates

        clat: xarray
            Latitude coordinates

        cell_area: xarray
            Array containing the area of all cells.

        Returns
        -------
        xarray
            Values of a variable in regular-lat-lon coordinates.
        """
        
        # estimate regular cell size based on grid cell area
        if cell_area is not None:
            cell_size = sqrt(np.mean(cell_area))
        else:
            # calculate mean distance for a number of selected points
            cartesian_coords = spherical2cartesian(clon, clat)
            kdtree = cKDTree(cartesian_coords)
            grid_point_distance, _ = kdtree.query(cartesian_coords[[0, len(cartesian_coords) // 2, len(cartesian_coords) - 1]], k=2)
            # assumption: equilateral triangle
            cell_size = np.sqrt(1.299 * grid_point_distance[:, 1].mean() ** 2)

        # calculate resolution based on estimated number of cells
        perimeter = 2 * np.pi * 6371229.0
        ncell = perimeter / cell_size
        resolution = 360.0 / ncell
        
        # define borders of the region
        lon0 = np.degrees(np.min(clon))
        lon1 = np.degrees(np.max(clon))
        lat0 = np.degrees(np.min(clat))
        lat1 = np.degrees(np.max(clat))
        
        # convert to regular-lon-lat coordinates 
        lon, lat = generate_coordinates(resolution, lon_range=[lon0, lon1], lat_range=[lat0, lat1])
        interpol = nearest_neighbour(clon, clat, lon, lat, src_grid="unstructured", dst_grid="regular")
        data_regular = interpol(data_icosahedral)
        return data_regular
    
    @classmethod
    def coords_name(cls, data):
        """
        Defines exact names of regular-lat-lon coordinates of a xarray object.

        Parameters
        ----------
        data : xarray
            Array contains values of variable and regular-lat-lon coordinates.

        Returns
        -------
        lon, lat: str
            Name of the longitude or latitude coordinate in the xarray.
        """
        
        # names for coordinates
        __lon_names = ["lon", "lons", "longitude", "longitudes", "clon", "rlon"]
        __lat_names = ["lat", "lats", "latitude", "latitudes", "clat", "rlat"]
        
        for dim_name in data.dims:
            if dim_name in __lon_names:
                lon = dim_name
            elif dim_name in __lat_names:
                lat = dim_name
                
        return lon, lat
    
    @classmethod
    def get_aspect_ratio(cls, lon0, lon1, lat0, lat1):
        """
        Defines an aspectratio of a figure based on a size of a region.

        Parameters
        ----------            
        lon0, lon1, lat0, lat1 : float
            Values of longitudes or latitudes of a defined region borders. 

        Returns
        -------
        float
            Aspectratio for a figure.
        """    
        
        lon_range = lon1 - lon0
        lat_range = lat1 - lat0
        aspectratio = lon_range / lat_range
        
        return aspectratio
        
    @classmethod
    def make_path_map(cls, feature, map_resolution):
        """
        Defines paths of coastlines or country borders.

        Parameters
        ----------
        feature : {'coastlines', 'borders'}
            Argument 'coastlines' for paths of coastlines. Argument 'borders' for paths of country borders.
            
        map_resolution : {'110m', '50m', '10m'}
            Definition of a map resolution.

        Returns
        -------
        xs, ys : list of lists
            Lists of lists contain separated X- or Y-coordinates of paths.
        """

        paths = []       
        # create paths
        if feature == 'coastlines':
            for geom in cartopy.feature.COASTLINE.with_scale(map_resolution).geometries():
                if geom.geom_type == 'LineString':
                    paths.append(list(geom.coords))
                else:
                    for g in geom.geoms:
                        paths.append(list(g.coords))
                    
        elif feature == 'borders':
            for geom in cartopy.feature.BORDERS.with_scale(map_resolution).geometries():
                if geom.geom_type == 'LineString':
                    paths.append(list(geom.coords))
                else:
                    for g in geom.geoms:
                        paths.append(list(g.coords))
                    
        else:
            raise ValueError(f"unsupported feature: {feature}")
               
        xs = []
        ys = []
        # separate X- and Y-coordinates of paths
        for i in range(len(paths)):
            coords = np.array([(vertex[0], vertex[1]) for vertex in paths[i]])
            lons = coords.T[0].tolist()
            lats = coords.T[1].tolist()
            xs.append(lons)
            ys.append(lats)

        return xs, ys
    
    @classmethod
    def get_contour_data(cls, x, y, z, colormap, filled_contours=True):
        """
        Gets parameters of contours created using Matplotlib.pyptlot. 

        Parameters
        ----------
        x, y : Numpy array
            Arrays contain X- or Y-coordinates of grid cells.
                      
        z : Numpy array
            2-dimensional array contains values of a variable for every grid cell.
                       
        colormap : str
            Name of a colormap.
            
        filled_contours : bool, optional, default=True
            Definition of contours type. Argument equals to True for filled contours and 
            equals to False for unfilled contours.
                      
        Returns
        -------
        xs, ys : list of lists
            Lists of lists contain separated X- or Y-coordinates of contours paths.
                
        colors : list of str
            List contains colors of contours in HEX format.
                
        values : list of str
            List contains values of contours.      
        """
        
        xs = []
        ys = []
        colors = []
        values = []
        
        # create contours using Matplotlib.pyptlot
        if filled_contours: 
            cs = plt.contourf(x, y, z, cmap=colormap)
        else:
            cs = plt.contour(x, y, z, cmap=colormap)
            
        # get colors and values of contours
        isolevelid = 0
        for isolevel in cs.collections:
            if filled_contours: 
                isocol = isolevel.get_facecolors()[0]
            else:
                isocol = isolevel.get_color()[0]  
            thecol = 3 * [None]
            theiso = str(cs.get_array()[isolevelid])
            isolevelid += 1

            for i in range(3):
                thecol[i] = int(255 * isocol[i])
            thecol = '#%02x%02x%02x' % (thecol[0], thecol[1], thecol[2])

            for path in isolevel.get_paths():
                colors.append(thecol)
                values.append(theiso)
                
        # get paths of contours        
        for i in range(len(cs.allsegs)):
            for j in range(len(cs.allsegs[i])):
                xs.append(cs.allsegs[i][j].T[0].tolist())
                ys.append(cs.allsegs[i][j].T[1].tolist())
                
        # close unclosed paths for a filled contours case   
        if filled_contours:
            # check which contours are closed
            closed_index = []
            for i in range(len(xs)):
                if xs[i][0] == xs[i][-1]:
                    closed_index.append(i)
            closed_index_np = np.asarray(closed_index)
            
            # rewrite contours to have only closed parts of contours
            xs_unclosed = xs
            ys_unclosed = ys
            xs = []
            ys = []
            for i in range(len(xs_unclosed)):
                # make no modifications for closed contours
                if i in closed_index_np:
                    xs.append(xs_unclosed[i])
                    ys.append(ys_unclosed[i])
                # cut off unclosed part of unclosed contours
                else:
                    index = []
                    for j in range(len(xs_unclosed[i])):
                        for l in range(j+1, len(xs_unclosed[i])):
                            if xs_unclosed[i][j] == xs_unclosed[i][l] and ys_unclosed[i][j] == ys_unclosed[i][l]:
                                index.append(j)
                                index.append(l-1)
                    index_np = np.asarray(index)
                    xs_closed = xs_unclosed[i][index_np[0]:index_np[1]]
                    ys_closed = ys_unclosed[i][index_np[0]:index_np[1]]
                    xs.append(xs_closed)
                    ys.append(ys_closed)
            
        plt.close()

        return xs, ys, colors, values
    
    @classmethod
    def generate_colorscale(cls, colors, values):
        """
        Generates ordered sets of unique colors and unique values.
    
        Parameters
        ----------
        colors : list of str
            Set of colors. List of HEX or RGB color strings.
                 
        values : list of str
            Set of values.

        Returns
        -------
        colorset : list of tuples
            Set of ordered unique colors. List containing tuples mapping a normalized color values. 
                 
        valueset : Numpy array
            Set of ordered unique float values.
                 
        tick_interval : float
            Value of intervals between ticks in a colorbar.
        """
        
        # select unique colors and generate a normalized colorset
        colors_unique = [list(OrderedDict.fromkeys(colors).keys())]
        
        colorset = []
        divisions = 1. / len(colors_unique) 
        c_index = 0.
        for cset in colors_unique:
            nsubs = len(cset)
            sub_divisions = divisions / nsubs
            for subcset in cset:
                colorset.append((c_index,subcset))
                colorset.append((c_index + sub_divisions - .001,subcset))
                c_index = c_index + sub_divisions
        colorset[-1] = (1, colorset[-1][1])
        
        # select unique values and get a tick interval
        valueset = np.unique(np.asarray(values, dtype=np.float32))
        tick_interval = valueset[1] - valueset[0]

        return colorset, valueset, tick_interval
    
    @classmethod
    def get_streamline_data(cls, x, y, u, v, density):
        """
        Gets paths of streamlines created using Matplotlib.pyplot.

        Parameters
        ----------
        x, y : xarray
            Arrays contain X- or Y- coordinates of grid cells.
                  
        u, v : xarray
            2-dimensional arrays contain values of horizontal or vertical components of a variable.
            
        density : float
            Controls the closeness of vectors. When density = 1, the domain is divided into a 30x30 grid.
              
        Returns
        -------
        Numpy array
            3-dimensional array contains X- and Y-coordinates of streamlines.
        """
        
        # interpolate over 2D grid
        lons = np.linspace(x.min(), x.max(), x.size)
        lats = np.linspace(y.min(), y.max(), y.size)
        
        # get paths of streamlines created by Matplotlib.pyplot
        stream = plt.streamplot(lons, lats, u, v, density=density)
        streamlines_path = np.asarray(stream.lines.get_segments()) 
    
        plt.close()
        
        return streamlines_path
        
    @classmethod
    def reduce_vector_data(cls, x, y, u, v, density):
        """
        Gets parameters for a reduced number of vectors.

        Parameters
        ----------
        x, y : xarray.DataArray
            Arrays contain X- or Y-coordinates of grid cells.

        u, v : xarray.DataArray
            2-dimensional arrays contain values of horizontal or vertical components of a variable.
            
        density : float
            Controls the closeness of vectors. When density = 1, the domain is divided into a 30x30 grid.

        Returns
        -------
        lons, lats : Numpy array
            2-dimensional array contain X- or Y-coordinates of a reduced number of grid cells.

        us, vs : Numpy array
            2-dimensional arrays contain reduced values of horizontal or vertical components of a variable.
        """
        
        # reduce a number of vectors
        nlon = int(density*30)
        nlat = int(density*30)
        if nlon > x.size and nlat > y.size:
            # no need for reduction
            lons, lats = np.meshgrid(x, y)
            return lons, lats, np.asarray(u), np.asarray(v)
        if nlon > x.size:
            nlon = x.size
        if nlat > y.size:
            nlat = y.size
        
        # create new arrays for a reduced number of vectors
        lon = np.zeros(nlon)
        lat = np.zeros(nlat)
        
        us = np.zeros((nlat, nlon))
        vs = np.zeros((nlat, nlon))
        
        # define intervals of a new grid
        dlon = int(floor(x.size / nlon))
        dlat = int(floor(y.size / nlat))
        dlon_half = int(floor(0.5*dlon))
        dlat_half = int(floor(0.5*dlat))
        
        # define a new grid
        for i in range(nlat):
            lat[i] = y[i*dlat + dlat_half]  
        for j in range(nlon):
            lon[j] = x[j*dlon + dlon_half]  
            
        lons,lats = np.meshgrid(lon, lat)

        # reduce a number of vectors, replacement by mean values
        for i in range(nlat):
            for j in range(nlon):
                u_temp = u[i*dlat:(i+1)*dlat, j*dlon:(j+1)*dlon]             
                v_temp = v[i*dlat:(i+1)*dlat, j*dlon:(j+1)*dlon]
                if np.all(np.isnan(u_temp)):
                    us[i,j] = np.nan
                    vs[i,j] = np.nan
                else:
                    us[i,j] = np.mean(u_temp)                           
                    vs[i,j] = np.mean(v_temp)

        return lons, lats, us, vs
    
    @classmethod
    def get_reference_arrow_data(cls, us, vs):
        """
        Gets parameters of a reference arrow.

        Parameters
        ----------
        us, vs : Numpy array
            2-dimensional arrays contain values of horizontal or vertical components of a variable.

        Returns
        -------                                  
        ref_arrow : int
            Value of a reference arrow.
        """
        
        # reference arrow = arrow with maximum length
        speed_max = np.nanmax(np.sqrt(us**2 + vs**2))

        # round and make a reference arrow multiple of 5            
        ref_arrow = int(np.round(speed_max/5.0)*5.0)
        
        return ref_arrow
    
    @classmethod
    @abstractmethod
    def make_colorbar(cls, colorset, valueset, tick_interval, variable_name, x_colorbar):
        """
        Creates a colorbar.

        Parameters
        ----------
        colorset : list of tuples
            Set of ordered unique colors. List containing tuples mapping a normalized value to a HEX or RGB. 
                 
        valueset : Numpy array
            Set of ordered unique double values.
                 
        tick_interval : float
            Value of intervals between ticks in a colorbar.
                      
        variable_name : str
            Name of a variable.
                      
        x_colorbar : float
            Position of a colorbar on axis X.

        Returns
        -------
        Plotly Scattergl or Bokeh ColorBar object
        """