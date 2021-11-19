from .backend import InteractiveBackend

import numpy as np

import matplotlib.pyplot as plt
from bokeh.models import FixedTicker, LinearColorMapper, ColorBar



class InteractiveBackendBokeh(InteractiveBackend):
    """
    Implementation of the interactive plotting backend for Bokeh.
    """
    
    @classmethod
    def make_colorbar(cls, colorset, valueset, tick_interval, variable_name, x_position=None):
        """
        Creates a colorbar for Bokeh.

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
                      
        x_position : float, optional, default=None
            Position of a colorbar on axis X.

        Returns
        -------
        Bokeh ColorBar object
        """
        
        # take only a colors definition without normalization
        colorset_modified = []
        for i in range(len(colorset)):
            colorset_modified.append(colorset[i][1])
            
        # define colorbar parameters    
        ticker = FixedTicker(ticks=valueset)
        color_mapper = LinearColorMapper(palette=colorset_modified, low=valueset[0], high=valueset[-1]+tick_interval)

        return ColorBar(color_mapper=color_mapper, ticker=ticker, label_standoff=12,
                        border_line_color=None, location=(0, 0), name=variable_name)

    @classmethod
    def separate_streamline_data(cls, streamlines_path):
        """
        Separates individual streamlines and also separates X- and Y- coordinates of them.

        Parameters
        ----------
        streamlines_path : array
            3-dimensional array contains X- and Y-coordinates of streamlines.
                  
        Returns
        -------
        xs, ys : list of lists
            Lists of lists contain X- or Y-coordinates of streamlines.
        """
        
        n = 0; k0 = 0
        k = []; xs = []; ys = []   
        
        # find beginning and ending indices for individual streamlines 
        for i in range(1,len(streamlines_path)):
            if streamlines_path[i-1,1,1] != streamlines_path[i,0,1] or streamlines_path[i-1,1,0] != streamlines_path[i,0,0]:
                n += 1
                k.append(i-1)
                
        # separate X- and Y- coordinates
        for ks in k:
            k1 = ks
            xsi = streamlines_path[k0:k1,1,0]
            ysi = streamlines_path[k0:k1,1,1]
            xs.append(xsi)
            ys.append(ysi)
            k0 = ks+1
    
        plt.close()
    
        return xs, ys
    
    @classmethod
    def separate_vector_data(cls, lons, lats, us, vs, scale, ref_arrow, lon0_map, lat1_map, lat1_map_arrow):
        """
        Separates parameters of vectors and a reference arrow. Creates arrowheads for vectors.

        Parameters
        ----------
        lons, lats : Numpy array
            2-dimensional arrays contain X- or Y-coordinates of a reduced number of grid cells.

        us, vs : Numpy array
            2-dimensional arrays contain values of horizontal or vertical components of a variable.
             
        scale : float, optional, default=0.02
            Possibility to scale a size of arrows(ideally to avoid overlap).
            
        ref_arrow : int
            Value of a reference arrow.
            
        lon0_map, lat1_map : float
            Longitude and latitude from region borders (left top corner) for definition of a reference arrow position.
            
        lat1_map_arrow : float
            Value equals approximately 10% of a height of a region. Used for definition of a reference arrow position for 2D plots.

        Returns
        -------
        xs, ys : list of arrays
            Lists of arrays contain X- or Y-coordinates of vectors and arrowheads.
                  
        xs_ref_arrow, ys_ref_arrow : list of arrays
            Lists of arrays contain X- or Y-coordinates of a reference arrow.
        """ 
        
        # define position of the reference arrow in Y-axis
        lat1_map = lat1_map - 0.8*lat1_map_arrow

        # define parameters of vectors
        speed = np.sqrt(us*us + vs*vs)
        theta = np.arctan(abs(vs/us))
        
        # define parameters for plotting
        x0 = lons.flatten()
        y0 = lats.flatten()
        
        length = speed.flatten()*scale
        angle = theta.flatten() 
        x1 = x0 + np.sign(us).flatten() * length * np.cos(angle)
        y1 = y0 + np.sign(vs).flatten() * length * np.sin(angle)
        
        # define arrowhead parameters
        x1_head = x1 - np.sign(us).flatten() * length/5 * np.cos(angle + np.radians(30))
        y1_head = y1 - np.sign(vs).flatten() * length/5 * np.sin(angle + np.radians(30))
        x2_head = x1 - np.sign(us).flatten() * length/5 * np.cos(angle - np.radians(30))
        y2_head = y1 - np.sign(vs).flatten() * length/5 * np.sin(angle - np.radians(30))
        
        # separate X- and Y-coordinates of vectors
        xs = []; ys = []  
        for i in range(len(x0)):
            xsi = np.asarray([x0[i], x1[i]])
            ysi = np.asarray([y0[i], y1[i]])
            xs.append(xsi)
            ys.append(ysi)
        # add arrowheads to a vector definition    
        for i in range(len(x0)):
            xsi_head = np.asarray([x1_head[i], x1[i], x2_head[i]])
            ysi_head = np.asarray([y1_head[i], y1[i], y2_head[i]])
            xs.append(xsi_head)
            ys.append(ysi_head)
        
        # define and separate reference arrow coordinates
        ref_arrow_length  = ref_arrow*scale
        
        xs_arrow_line = np.asarray([lon0_map+0.5, lon0_map+0.5+ref_arrow_length])
        ys_arrow_line = np.asarray([lat1_map, lat1_map])
            
        x1_arrow_head = xs_arrow_line[1] - ref_arrow_length/5 * np.cos(np.radians(30))
        y1_arrow_head = ys_arrow_line[1] - ref_arrow_length/5 * np.sin(np.radians(30))
        x2_arrow_head = xs_arrow_line[1] - ref_arrow_length/5 * np.cos(-np.radians(30))
        y2_arrow_head = ys_arrow_line[1] - ref_arrow_length/5 * np.sin(-np.radians(30))
            
        xs_arrow_head = np.asarray([x1_arrow_head, xs_arrow_line[1], x2_arrow_head])
        ys_arrow_head = np.asarray([y1_arrow_head, ys_arrow_line[1], y2_arrow_head])
            
        xs_ref_arrow = []; ys_ref_arrow = []
        xs_ref_arrow.append(xs_arrow_line); xs_ref_arrow.append(xs_arrow_head)
        ys_ref_arrow.append(ys_arrow_line); ys_ref_arrow.append(ys_arrow_head)
        
        return xs, ys, xs_ref_arrow, ys_ref_arrow