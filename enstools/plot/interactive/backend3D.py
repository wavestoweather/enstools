from .backend import InteractiveBackend
from .backend import Function, Stream

from enstools.misc import generate_coordinates
from enstools.interpolation import nearest_neighbour

from numba import jit
import numpy as np
from math import sqrt, floor
from collections import OrderedDict
from scipy.interpolate import interp2d

try:
    import cartopy
except ModuleNotFoundError:
    raise AssertionError("This feature requires cartopy which is not installed.")
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import plotly.graph_objs as go



class InteractiveBackend3D(InteractiveBackend):
    """
    Abstract backend class that will be implemented for different plotting backends. Functions that are used for all
    backends can be implemented directly here.
    """
    
    @classmethod
    def spherical_to_cartesian(cls, lons, lats, radius = 6371229.0):
        """
        Transforms spherical coordinates into 3D cartesian coordinates.
    
        Parameters
        ----------
        lons, lats : Numpy array
            Numpy arrays contain longitudes or latitudes in radians.
                 
        radius : float, optional, default=6371229.0
            Radius of a sphere for a transformation. For default an argument equals to Earth radius.
                
        Returns
        -------
        Numpy array
            3-dimensional Numpy array contains cartesian coordinates.
        """
            
        cartesian = np.empty((lons.shape[0], 3))
        lon = lons + np.pi
        lat = lats - np.pi / 2.0
        cartesian[:, 0] = radius * np.sin(lat) * np.cos(lon)
        cartesian[:, 1] = radius * np.sin(lat) * np.sin(lon)
        cartesian[:, 2] = radius * np.cos(lat)

        return cartesian
    
    @classmethod
    def separate_cartesian(cls, cartesian):  
        """
        Separates cartesian coordinates into X, Y and Z components.

        Parameters
        ----------
        cartesian: array
            3D array contains cartesian coordinates.
                      
        Returns
        -------
        xs, ys, zs : list of lists
            Lists of lists contain X-, Y- or Z- component of cartesian coordinates in radians.    
        """
        
        # separate X-, Y- and Z-coordinates
        xs = []
        ys = []
        zs = []
        for i in range(len(cartesian)):
            x_i = []
            y_i = []
            z_i = []
            for j in range(len(cartesian[i])):
                x_i.append(cartesian[i][j][0])
                y_i.append(cartesian[i][j][1])
                z_i.append(cartesian[i][j][2])
            xs.append(x_i)
            ys.append(y_i)
            zs.append(z_i)   
                                
        return xs, ys, zs
       
    @classmethod
    def paths_cartesian(cls, xs, ys, radius = 6381229.0):
        """
        Transforms paths of lines in degrees into paths in Cartesian coordinates leaving initial paths structure.

        Parameters
        ----------
        xs, ys : list of lists
            Lists of lists contain X- or Y-coordinates of paths.
            
        radius : float, optional, default=6371229.0
            Radius of a sphere for a transformation. For default an argument equals to Earth radius.          
                      
        Returns
        -------
        xs, ys, zs : list of lists
            Lists of lists contain X-, Y- or Z- component of cartesian coordinates of paths.                
        """
               
        # transform to cartesian coordinates
        xs_radians = np.array([np.radians(np.array(xi)) for xi in xs], dtype=object)
        ys_radians = np.array([np.radians(np.array(yi)) for yi in ys], dtype=object)
        
        cartesian = np.array([cls.spherical_to_cartesian(xs_radians[i], ys_radians[i], radius=radius) 
                              for i in np.arange(len(xs_radians))], dtype=object)     
        npaths = len(cartesian)
        
        # separate X-, Y- and Z-coordinates
        xs, ys, zs = cls.separate_cartesian(cartesian)  
                    
        return xs, ys, zs, npaths
    
    @staticmethod
    @jit(nopython=True)
    def point_in_region(lons, lats, x, y, cell_size):
        """
        Checks if point is within a region.
        
        Parameters
        ----------                           
        lons, lats : Numpy array
            Numpy arrays contain longitudes or latitudes of cell vertices in radians.
            
        x, y : float
            Values of longitude or latitude of the point in degrees. 
            
        cell_size : float
            Approximate size of a grid cell.

        Returns
        -------
        bool : 
            Equals to True if the point is within the region.
        """
        
        lons = np.degrees(lons)
        lats = np.degrees(lats)
        
        degrees_to_meters = 2 * np.pi * 6371229.0 / 360
        for i in range(len(lons)):
            in_region = False
            distance = (lons[i] - x)*(lons[i] - x) + (lats[i] - y)*(lats[i] - y)
            if degrees_to_meters*sqrt(distance) <= cell_size:
                in_region = True
                break
                
        return in_region

    @classmethod
    def make_path_map3D(cls, feature, cell_size, lons, lats, regional, lon0, lon1, lat0, lat1, map_resolution):
        """
        Defines paths of coastlines or country borders for a defined region. Includes selection of a defined region. 
        Adopted for a 3D plotting.

        Parameters
        ----------
        feature : {'coastlines', 'borders'}
            Argument 'coastlines' for paths of coastlines. Argument 'borders' for paths of country borders.
            
        cell_size : float
            Approximate size of a grid cell.
                
        lons, lats : Numpy array
            Numpy arrays contain longitudes or latitudes in radians.
            
        regional : bool
            Argument defines necessity to select part of paths for a defined region. Equals to True for a regional plot.
            
        lon0, lon1, lat0, lat1 : float
            Values of longitudes or latitudes of a defined region borders. 
            
        map_resolution : {'110m', '50m', '10m'}
            Definition of a map resolution.

        Returns
        -------
        xs_map, ys_map, zs_map : list of lists
            Lists of lists contain separated X-, Y- or Z-coordinates of paths in a defined region.
            
        npaths : int
            Number of paths
        """
        
        paths = []
        # create paths
        if feature == 'coastlines':
            for geom in cartopy.feature.COASTLINE.with_scale(map_resolution).geometries():
                if geom.geom_type == 'LineString':
                    paths.append(list(geom.coords))
                else:
                    for g in geom:
                        paths.append(list(g.coords))
                    
        elif feature == 'borders':
            for geom in cartopy.feature.BORDERS.with_scale(map_resolution).geometries():
                if geom.geom_type == 'LineString':
                    paths.append(list(geom.coords))
                else:
                    for g in geom:
                        paths.append(list(g.coords))
                    
        else:
            raise ValueError(f"unsupported feature: {feature}")
        
        # select paths which are in a defined region and separate X- and Y-coordinates of paths  
        # preselect paths with are in a square frame
        xs_frame = []
        ys_frame = []        
        for i in range(len(paths)):
            xs_i = []
            ys_i = []
            in_frame = True
            prev_step_in_frame = True
            for j in range(len(paths[i])):
                in_frame = paths[i][j][0]>=lon0 and paths[i][j][0]<=lon1 and paths[i][j][1]>=lat0 and paths[i][j][1]<=lat1
                if prev_step_in_frame:
                    if in_frame:
                        xs_i.append(paths[i][j][0])
                        ys_i.append(paths[i][j][1])
                        if j == len(paths[i])-1:
                            xs_frame.append(xs_i)
                            ys_frame.append(ys_i)
                    elif len(xs_i) != 0:
                        xs_frame.append(xs_i)
                        ys_frame.append(ys_i)
                elif in_frame:
                    xs_i = []
                    ys_i = []
                    xs_i.append(paths[i][j][0])
                    ys_i.append(paths[i][j][1])
                prev_step_in_frame = in_frame
        
        # final (more accurate) selection and transformation to degrees
        if regional:
            xs_degrees = []
            ys_degrees = []  
        
            for i in range(len(xs_frame)):
                xs_i = []
                ys_i = []
                for j in range(len(xs_frame[i])):
                    in_region = cls.point_in_region(lons, lats, xs_frame[i][j], ys_frame[i][j], cell_size)
                    if in_region:
                        xs_i.append(xs_frame[i][j])
                        ys_i.append(ys_frame[i][j])
                    elif len(xs_i) != 0:
                        xs_degrees.append(xs_i)
                        ys_degrees.append(ys_i)
                        xs_i = []
                        ys_i = [] 
                if in_region and len(xs_i) != 0:
                    xs_degrees.append(xs_i)
                    ys_degrees.append(ys_i)
        else:
            xs_degrees = xs_frame
            ys_degrees = ys_frame
        
        # transform to cartesian coordinates and separate coordinates
        xs_map, ys_map, zs_map, npaths = cls.paths_cartesian(xs_degrees, ys_degrees, radius=6381229.0)
        
        return xs_map, ys_map, zs_map, npaths
    
    @classmethod
    def set_aspectratio3D(cls, xs, ys, zs):
        """
        Defines an aspectratio of a figure based ob size of a region.

        Parameters
        ----------            
        xs, ys, zs : float
            Values of X-, Y- or Z-coordinates of a defined region. 

        Returns
        -------
        xratio, yratio, zratio
            Aspectratio for a 3D figure.
        """    
                  
        # get exices ranges          
        dx = abs(np.max(xs) - np.min(xs))
        dy = abs(np.max(ys) - np.min(ys))
        dz = abs(np.max(zs) - np.min(zs))
        norma = np.max(np.array([dx, dy, dz]))
        xratio = dx / norma
        yratio = dy / norma
        zratio = dz / norma
        
        return xratio, yratio, zratio  
    
    @classmethod    
    def set_colors(cls, data, colormap, function):
        """
        Defines colors of individual grid cells.
    
        Parameters
        ----------
        data : xarray
            Values of a variable and a range of other parameters describing a variable and a grid.
                 
        colormap : str
            Name of a colormap.
        
        function : Enum element
            Definition of a Plotly function used for a plotting. Element from Function Enum.

        Returns
        -------
        list
            List contains colors for every grid cell based on a defined colormap. 
        """
        
        # define colormap
        cmap = cm.get_cmap(colormap)
        # normalize values of a variable
        normalization = Normalize(vmin=np.min(data[~np.isnan(data)]), vmax=np.max(data[~np.isnan(data)]))
        normalized_data = normalization(data)
        # set colors 
        data_cmap = cmap(normalized_data)
        
        # create list of colors in a right format for plotting
        colors = []
        for i in range(len(data_cmap)):
            if function == Function.TRISURF:
                if data_cmap[i][0] == data_cmap[i][1] == data_cmap[i][2] == data_cmap[i][3] == 0:
                    colors.append(tuple((255,255,255,0)))
                else:
                    colors.append(tuple(data_cmap[i]))
            elif function == Function.MESH3D:
                colors.append(data_cmap[i])

        return colors
    
    @classmethod
    def make_colorbar(cls, colorset, valueset, tick_interval, variable_name, x_colorbar):
        """
        Creates a colorbar for Plotly.

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
        Plotly Scattergl object
            X- and Y-coordinates of a Scattergl object are not define. The object contains only a colorbar. 
        """

        return go.Scattergl(x=[None], y=[None], mode='markers', showlegend=False, legendgroup=variable_name,
                            marker=dict(colorscale=colorset, cmin=valueset[0], cmax=valueset[-1]+tick_interval,
                                        colorbar=dict(tickvals=valueset, outlinecolor='black', outlinewidth=0.5, x=x_colorbar)))
    
    @classmethod
    def make_colorbar_continuous(cls, data, colormap):
        """
        Creates a continouos colorbar.
    
        Parameters
        ----------
        data : xarray
            Values of a variable and a range of other parameters describing a variable and a grid.
                 
        colormap : str
            Name of a colormap.
            
        Returns
        -------
        Plotly Scattergl object
            X- and Y-coordinates of a Scattergl object are not define. The object contains only a colorbar.
        """      
        
        return go.Scattergl(x=[None],y=[None], mode='markers', showlegend=False, 
                            marker=dict(colorscale=colormap, showscale=True, cmin=np.min(data), cmax=np.max(data), 
                                        colorbar=dict(outlinecolor='black', outlinewidth=0.5)), hoverinfo='none')
    
    @classmethod
    def set_camera_eye(cls, lons, lats):
        """
        Defines a camera eye. 
    
        Parameters
        ----------
        lons, lats : Numpy array
            Numpy arrays contain longitudes or latitudes in radians.
            
        Returns
        -------
        Numpy array
            Normal vector at a center point of a region.  
        """        
        
        # define two surfaces, one above other 
        cartesian_coordinates = cls.spherical_to_cartesian(lons, lats, radius=6371229.0)
        xs = cartesian_coordinates.T[0]
        ys = cartesian_coordinates.T[1]
        zs = cartesian_coordinates.T[2]
        
        cartesian_coordinates_above = cls.spherical_to_cartesian(lons, lats, radius=6371230.0)
        xs_above = cartesian_coordinates_above.T[0]
        ys_above = cartesian_coordinates_above.T[1]
        zs_above = cartesian_coordinates_above.T[2]
        
        # define central points of surfaces
        x_mean = np.mean(xs)
        y_mean = np.mean(ys)
        z_mean = np.mean(zs)
        center_point = np.array([x_mean, y_mean, z_mean])
        
        x_mean_above = np.mean(xs_above)
        y_mean_above = np.mean(ys_above)
        z_mean_above = np.mean(zs_above)
        center_point_above = np.array([x_mean_above, y_mean_above, z_mean_above])
        
        # calculate normal vector at a central point 
        normal = center_point_above - center_point

        return normal