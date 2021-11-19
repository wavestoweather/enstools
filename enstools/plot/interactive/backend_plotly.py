from .backend import InteractiveBackend

import numpy as np

import matplotlib.pyplot as plt
import plotly.graph_objs as go



class InteractiveBackendPlotly(InteractiveBackend):
    """
    Implementation of the interactive plotting backend for Plotly.
    """
    
    @classmethod
    def make_scattergl_map(cls, xs, ys):
        """
        Makes a Plotly Scattergl object for coastlines and country borders.

        Parameters
        ----------
        xs, ys : list of lists
            Lists of lists contain separated X- or Y-coordinates of paths.

        Returns
        -------
        list
            List contains a Plotly Scattergl objects. 
        """
        
        traces = []
        for i in range(len(xs)):
            traces.append(go.Scattergl(x=xs[i], y=ys[i], mode='lines', line=go.scattergl.Line(color="black", width=0.7),
                                       name='map', showlegend=False, opacity=0.7))
        return traces
    
    @classmethod
    def make_filled_contour(cls, x, y, z, variable_name, colormap, x_colorbar):
        """
        Creates filled contours using specialized Plotly function.

        Parameters
        ----------
        x, y : array
            Arrays contain X- or Y-coordinates of grid cells.
                      
        z : array
            2-dimensional array contain values of a variable for every grid cell.
                      
        variable_name : str
            Name of a variable.
                       
        colormap : str
            Name of a colormap.
                      
        x_colorbar : float
            Position of a colorbar on axis X.

        Returns
        -------
        Plotly Contour object
            Filled contours with a colorbar. 
        """

        return go.Contour(z=z, x=x, y=y, showlegend=True, name=variable_name,
                          autocontour=True, ncontours=10, showscale=True, colorscale=colormap,
                          colorbar=dict(outlinecolor='black', outlinewidth=0.5, bordercolor='white', x=x_colorbar))

    @classmethod
    def make_unfilled_contour(cls, xs, ys, color, value, variable_name, showlegend=True):
        """
        Creates a Scattergl object for an unfilled contour.

        Parameters
        ----------
        xs, ys : list
            Lists contain X- or Y-coordinates of a contour.
                      
        color : str
            Color of a contour.
                       
        value : str
            Value of a contour.
                      
        variable_name : str
            Name of a variable.
                       
        showlegend : bool, default=True
            Definition of displaying of a contour at a legend.

        Returns
        -------
        Plotly Scattergl object
            Scattergl object contains an unfilled contour.
        """ 

        return go.Scattergl(x=xs, y=ys, mode='lines', line=go.scattergl.Line(color=color, width=1),
                            showlegend=showlegend, legendgroup=variable_name, name=value)
    
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
                 
        tick_interval : double
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
    def make_scattergl_streamline(cls, streamlines_path, color, showlegend=True):
        """
        Makes a Scattergl object for streamlines.

        Parameters
        ----------
        streamlines_path : array
            3-dimensional array contains X- and Y-coordinates of streamlines.
            
        color : str
            Defines a color of streamlines.
        
        showlegend : bool, optional, default=True
            Definition of displaying of streamlines at a legend.

        Returns
        -------
        Plotly Scattergl object                           
        """

        return go.Scattergl(x=streamlines_path.T[0], y=streamlines_path.T[1],
                            mode='lines', line=go.scattergl.Line(color=color, width=1), 
                            showlegend=showlegend, legendgroup='streamlines', name='streamlines')
   