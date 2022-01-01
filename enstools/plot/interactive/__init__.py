import logging
import warnings
import numpy as np
from math import sqrt
from enstools.plot.core import get_coordinates_from_xarray

from .backend import InteractiveBackend, Backend, Stream, Function
from .backend_plotly import InteractiveBackendPlotly
from .backend_bokeh import InteractiveBackendBokeh
from .backend3D import InteractiveBackend3D

import plotly.graph_objs as go
import plotly.figure_factory as ff 

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Legend

from ...core import check_arguments


@check_arguments(dims={'variable': ('lat', 'lon')})
def interactive_contours(variable, lon=None, lat=None, **kwargs):
    """
    Creates a plot of interactive contours.

    Parameters
    ----------
    variable : xarray
            Values of a variable and a range of other parameters describing the variable and the grid.

    lon : xarray.DataArray or np.ndarray or str
            longitude coordinate or name of longitude coordinate. The name may only be used for xarray variables.

    lat : xarray.DataArray or np.ndarray or str
            latitude coordinate or name of longitude coordinate. The name may only be used for xarray variables.


    Other optional keyword arguments:

    **kwargs
            *figure*: bokeh or plotly figure object
                If provided, this figure instance will be used (and returned), otherwise a new
                figure will be created.

            *filled*: [*True* | *False*]
                If True a filled contour is plotted, which is the default

            *selected_backend* : Enum element, optional, default=Library.BOKEH
                Definition of a library for plotting. Element from Library Enum.
                 
            *aspect* : float, optional, default=None
                Aspect ratio between a figure width and a height as a fraction.
        
            *size* : int, optional, default=800
                Size of a figure width in pixels. Size of a figure height is calculated from an aspectratio.
        
            *map_resolution* : {'110m', '50m', '10m'}, optional, default='110m'
                Definition of a map resolution.

            *cmap* : str
                Name of the color map to use. Default: CMRmap_r

            *line_width* : int
                Line width, default is 1.
                      
    Returns
    -------
    figure
        Figure object contains interactive contours with a colorbars.            
    """

    # initialize a backend object based on the backend argument
    selected_backend = kwargs.pop("backend", Backend.BOKEH)
    if selected_backend == Backend.PLOTLY:
        _backend = InteractiveBackendPlotly()       
    elif selected_backend == Backend.BOKEH:
        _backend = InteractiveBackendBokeh()        
    else:
        raise ValueError(f"unsupported backend: {selected_backend}! backend must be an instance of Library Enum: PLOTLY or BOKEH.")
    
    # add warnings
    aspect_ratio = kwargs.pop("aspect", None)
    size = kwargs.pop("size", 800)
    previous_figure = kwargs.pop("figure", None)
    if previous_figure is not None:
        if aspect_ratio is not None or size != 800:
            warnings.warn("Change of figure size parameters is impossible for a new figure. Figure size parameters stay the same as in a previous figure.")

    map_resolution = kwargs.pop("map_resolution", '110m')
    if map_resolution not in {'110m', '50m', '10m'}:
        raise ValueError(f"unsupported map resolution: {map_resolution}! map_resolution must be an instance of: '110m', '50m' or '10m'.")
        
    # get the coordinates
    lon, lat = get_coordinates_from_xarray(variable, lon, lat, rad2deg=False)

    # interpolate to regular lat-lon grid if on icosahedral
    if variable.ndim == 1 and lon.ndim == 1 and lat.ndim == 1:
        variable = _backend.icosahedral_to_regular_coords(variable, clon=lon, clat=lat)
        lon, lat = get_coordinates_from_xarray(variable, create_mesh=False)

    # create a plot with filled contour lines?
    filled_contours = kwargs.pop('filled', True)

    # define figure parameters
    x_colorbar = None    # colorbar position on X-axis
    default_cmap = "CMRmap_r"
    if selected_backend == Backend.PLOTLY and filled_contours:
        default_cmap = "Inferno_r"
    colormap = kwargs.pop("cmap", default_cmap)
    if previous_figure is not None:
        fig = previous_figure
        x_colorbar = 1.2
        colormap = "Greys"

    variable_name = variable._name
        
    # define map borders
    lon0_map = float(lon.min())
    lon1_map = float(lon.max())
    lat0_map = float(lat.min())
    lat1_map = float(lat.max())

    # define an aspect ratio
    if aspect_ratio is None:
        aspect_ratio = _backend.get_aspect_ratio(lon0_map, lon1_map, lat0_map, lat1_map)

    # make paths for coastines and country borders
    xs_coas, ys_coas = _backend.make_path_map('coastlines', map_resolution)
    xs_bord, ys_bord = _backend.make_path_map('borders', map_resolution)

    # width for contour lines
    line_width = kwargs.pop("line_width", 1)

    # define contours and its colorscale for unfilled contours
    if not filled_contours:
        xs, ys, colors, values = _backend.get_contour_data(lon, lat, variable,
                                                           colormap, filled_contours=filled_contours)
        colorset, valueset, tick_interval = _backend.generate_colorscale(colors, values)

    if selected_backend == Backend.PLOTLY:
            
        # create a Plotly objects for coastlines and country borders
        traces_coas = _backend.make_scattergl_map(xs_coas, ys_coas)
        traces_bord = _backend.make_scattergl_map(xs_bord, ys_bord)
        base = traces_coas + traces_bord
            
        # define figure paramerets
        if previous_figure == None:
            fig = go.Figure(data=base)
            fig.update_layout(autosize=False, width=size, height=size / aspect_ratio,
                              xaxis_title="lon", yaxis_title="lat", plot_bgcolor="#FFFFFF",
                              xaxis=dict(range=[lon0_map, lon1_map], showgrid=False),
                              yaxis=dict(range=[lat0_map, lat1_map], showgrid=False),
                              legend=dict(orientation="h", bordercolor="#E0E0E0", borderwidth=1, 
                                          yanchor="top", y=1.15, xanchor="left", x=0.01))
            
        if filled_contours:
            
            # define and add filled contours to a figure
            contours = _backend.make_filled_contour(lon, lat, variable, variable_name, colormap, x_colorbar)
            fig.add_trace(contours)
            
        else:
            
            # add unfilled contours to a figure
            fig.add_trace(_backend.make_unfilled_contour(xs[0], ys[0], colors[0], variable_name, 
                                                         variable_name, showlegend=True))
            for i in range(1,len(xs)):
                fig.add_trace(_backend.make_unfilled_contour(xs[i], ys[i], colors[i], values[i], 
                                                             variable_name, showlegend=False))
            # define and add a colorbar to a figure    
            colorbar = _backend.make_colorbar(colorset, valueset, tick_interval, variable_name, x_colorbar)
            fig.add_trace(colorbar)

    elif selected_backend == Backend.BOKEH:
            
        # define figure parameters
        if previous_figure is None:
            fig = figure(aspect_ratio=aspect_ratio, plot_width=size,
                         x_range=(lon0_map, lon1_map), y_range=(lat0_map, lat1_map),
                         x_axis_label="lon", y_axis_label="lat", output_backend="webgl")
            fig.xgrid.visible = False
            fig.ygrid.visible = False
            fig.outline_line_color = None 
            
        if filled_contours:
            
            # define and add filled contours to a figure
            xs, ys, colors, values = _backend.get_contour_data(lon, lat, variable, colormap, filled_contours=filled_contours)
            source = ColumnDataSource(data={'xs': xs, 'ys': ys, 'line_color': colors})
            contours = fig.patches(xs='xs', ys='ys', alpha=1, line_width=0,
                                   fill_color='line_color', line_color='line_color', source=source)
            # define a colorbar
            colorset, valueset, tick_interval = _backend.generate_colorscale(colors, values)
            colorbar = _backend.make_colorbar(colorset, valueset, tick_interval, variable_name)

        else:
            
            # define and add unfilled contours to a figure
            source = ColumnDataSource(data={'xs': xs, 'ys': ys, 'line_color': colors})
            contours = fig.multi_line(xs='xs', ys='ys', line_color='line_color', line_width=line_width, source=source)
            
            # define a colorbar
            colorbar = _backend.make_colorbar(colorset, valueset, tick_interval, variable_name)

        # add coaslines and country borders to a figure
        fig.multi_line(xs_coas, ys_coas, line_color="grey")
        fig.multi_line(xs_bord, ys_bord, line_color="grey")
            
        # define figure layout parameters
        legend = Legend(items=[(variable_name , [contours])], location="top_left")
        fig.add_layout(colorbar, "right")
        fig.add_layout(legend, "above")
        fig.legend.click_policy = "hide"

    # check if all keyword arguments have been used.
    if len(kwargs) > 0:
        logging.warning("unknown arguments: %s" % ",".join(kwargs.keys()))
    return fig


@check_arguments(dims={'variable_u': ('lat', 'lon'), 'variable_v': ('lat', 'lon')})
def interactive_streamlines(variable_u, variable_v, lon=None, lat=None, **kwargs):
    """
    Creates a plot of interactive streamlines or vectors.

    Parameters
    ----------
    variable_u, variable_v : xarray.DataArray
        u- and v-component of streamline data.
        
    lon : xarray.DataArray or np.ndarray or str
            longitude coordinate or name of longitude coordinate. The name may only be used for xarray variables.

    lat : xarray.DataArray or np.ndarray or str
            latitude coordinate or name of longitude coordinate. The name may only be used for xarray variables.


    Other optional keyword arguments:

    **kwargs
            *figure*: bokeh or plotly figure object
                If provided, this figure instance will be used (and returned), otherwise a new
                figure will be created.
              
            *selected_backend* : Enum element, optional, default=Library.BOKEH
                Definition of a library for plotting. Element from Library Enum.

            *aspect* : float, optional, default=None
                Aspect ratio between a figure width and a height as a fraction.

            *size* : int, optional, default=800
                Size of a figure width in pixels. Size of a figure height is calculated from an aspectratio.

            *map_resolution* : {'110m', '50m', '10m'}, optional, default='110m'
                Definition of a map resolution.

            *line_type* : Enum element, optional, default=Stream.LINE
                Definition of a line type. Element from Stream Enum.
        
            *density* : float, optional, default=2
                Definition of a closeness of streamlines. When density = 1, the domain is divided into a 30x30 grid.

            *scale* : float, optional, default=0.02
                Possibility to scale a size of arrows(for example, to avoid overlap).
                      
    Returns
    -------
    figure
        Figure object contains interactive streamlines or vectors.              
    """
    
    # initialize a backend object based on the backend argument
    selected_backend = kwargs.pop("backend", Backend.BOKEH)
    if selected_backend == Backend.PLOTLY:
        _backend = InteractiveBackendPlotly()
    elif selected_backend == Backend.BOKEH:
        _backend = InteractiveBackendBokeh()
    else:
        raise ValueError(f"unsupported backend: {selected_backend}! backend must be an instance of Library Enum: PLOTLY or BOKEH.")
    
    # add warnings
    aspect_ratio = kwargs.pop("aspect", None)
    size = kwargs.pop("size", 800)
    previous_figure = kwargs.pop("figure", None)
    map_resolution = kwargs.pop("map_resolution", '110m')
    if previous_figure is not None:
        fig = previous_figure
        if aspect_ratio is not None or size != 800 or map_resolution != '110m':
            warnings.warn("Change of figure size parameters is impossible for a new figure. Figure size parameters stay the same as in a previous figure.")
            
    if map_resolution not in {'110m', '50m', '10m'}:
        raise ValueError(f"unsupported map resolution: {map_resolution}! map resolution must be an instance of: '110m', '50m' or '10m'.")
        
    # get the coordinates
    lon, lat = get_coordinates_from_xarray(variable_u, lon, lat, rad2deg=False)

    # interpolate to regular lat-lon grid if on icosahedral
    if variable_u.ndim == 1 and lon.ndim == 1 and lat.ndim == 1:
        variable_u = _backend.icosahedral_to_regular_coords(variable_u, clon=lon, clat=lat)
        variable_v = _backend.icosahedral_to_regular_coords(variable_v, clon=lon, clat=lat)
        lon, lat = get_coordinates_from_xarray(variable_u, create_mesh=False)

    # define figure parameters
    color = "black"

    # define map borders
    lon0_map = float(lon.min())
    lon1_map = float(lon.max())
    lat0_map = float(lat.min())
    lat1_map = float(lat.max())
    
    # define an aspectratio
    if aspect_ratio is None:
        aspect_ratio = _backend.get_aspect_ratio(lon0_map, lon1_map, lat0_map, lat1_map)
    
    # make paths for coastines and country borders
    xs_coas, ys_coas = _backend.make_path_map('coastlines', map_resolution)
    xs_bord, ys_bord = _backend.make_path_map('borders', map_resolution)
    
    # define streamlines paths
    line_type = kwargs.pop("line_type", Stream.LINE)
    density = kwargs.pop("density", 2)
    scale = kwargs.pop("scale", 0.02)
    if line_type == Stream.LINE:
        streamlines = _backend.get_streamline_data(lon, lat, variable_u, variable_v, density)
        
    # reduce a number of grid cells for vectors and get reference arrow data
    elif line_type == Stream.VECTOR:
        lons, lats, us, vs = _backend.reduce_vector_data(lon, lat, variable_u, variable_v, density)

        # increase range of Y-axis for placing of a reference arrow 
        lat1_map_arrow = abs(0.1*(lat1_map-lat0_map))
        lat1_map = lat1_map + lat1_map_arrow
        
        ref_arrow = _backend.get_reference_arrow_data(us, vs)
        if hasattr(variable_u, 'units'):
            ref_arrow_string = (str(ref_arrow) + ' ' + variable_u.units)
        else:
            ref_arrow_string = str(ref_arrow)
        
    else:
        raise ValueError(f"unsupported lines type: {line_type}! line type must be an instance of Stream Enum: LINE or VECTOR.")
            
    if selected_backend == Backend.PLOTLY:
        # create a Plotly objects for coastlines and contry borders
        traces_coas = _backend.make_scattergl_map(xs_coas, ys_coas)
        traces_bord = _backend.make_scattergl_map(xs_bord, ys_bord)
        base = traces_coas + traces_bord
            
        # define figure parameters
        if previous_figure is None:
            fig = go.Figure(data=base)
            fig.update_layout(autosize=False, width=size, height=size/aspect_ratio,
                              xaxis_title="lon", yaxis_title="lat", plot_bgcolor="#FFFFFF", 
                              xaxis=dict(range=[lon0_map, lon1_map], showgrid=False), 
                              yaxis=dict(range=[lat0_map, lat1_map], showgrid=False), 
                              legend=dict(orientation="h", bordercolor='#E0E0E0', borderwidth=1, 
                                          yanchor="top", y=1.20, xanchor="left", x=0.01))
            
        if line_type == Stream.LINE:
            # add streamlines to a figure   
            fig.add_trace(_backend.make_scattergl_streamline(streamlines[0], color, showlegend=True))
            for i in range(len(streamlines)):
                fig.add_trace(_backend.make_scattergl_streamline(streamlines[i], color, showlegend=False))
                
        elif line_type == Stream.VECTOR:
            # define a reference arrow
            fig_ref_arrow_temp = (ff.create_quiver([lon0_map + 0.5], [lat1_map - 0.8*lat1_map_arrow], [ref_arrow], [0], 
                                                   scale=scale, arrow_scale=0.3, line_color=color, 
                                                   name="reference arrow", legendgroup="vectors", showlegend=False))
            
            # define vectors
            fig_temp = (ff.create_quiver(lons, lats, us, vs, scale=scale, arrow_scale=0.3, line_color=color,
                                         name="vectors", legendgroup="vectors"))
            
            # add vectors and a reference arrow to a figure
            fig.add_traces(data = fig_temp.data)
            fig.add_traces(data = fig_ref_arrow_temp.data)
            fig.add_trace(go.Scatter(x=[lon0_map + 0.5], y=[lat1_map - 0.8*lat1_map_arrow],
                                     mode="text", text=ref_arrow_string, textposition="top right",
                                     name="reference arrow", legendgroup="vectors", showlegend=False))
            
    elif selected_backend == Backend.BOKEH:
        # define figure parameters
        if previous_figure is None:
            fig = figure(aspect_ratio=aspect_ratio, plot_width=size,
                         x_range=(lon0_map, lon1_map), y_range=(lat0_map, lat1_map), 
                         x_axis_label="lon", y_axis_label="lat", output_backend="webgl")
            fig.xgrid.visible = False
            fig.ygrid.visible = False
            fig.outline_line_color = None      
        
        if line_type == Stream.LINE:
            # separate and add streamlines to a figure
            xs, ys = _backend.separate_streamline_data(streamlines)
            
            lines = fig.multi_line(xs, ys, color=color, line_width=2, line_alpha=0.8) 
            
            legend = Legend(items=[("streamlines" , [lines])], location="top_left")
        
        elif line_type == Stream.VECTOR:
            # define and add vectors and a reference arrow to a figure
            xs, ys, xs_ref_arrow, ys_ref_arrow = _backend.separate_vector_data(lons, lats, us, vs, scale, 
                                                                               ref_arrow, lon0_map, lat1_map, lat1_map_arrow)  
            
            lines = fig.multi_line(xs, ys, color=color, line_width=2)
            
            ref_arrow_text = fig.text(x=[lon0_map+0.5], y=[lat1_map-0.8*lat1_map_arrow], text=[ref_arrow_string])
            ref_arrow = fig.multi_line(xs_ref_arrow, ys_ref_arrow, color=color, line_width=2)
                                                      
            legend = Legend(items=[("vectors" , [lines, ref_arrow_text, ref_arrow])], location="top_left")     
            
        # add coastlines and country borders to a figure
        fig.multi_line(xs_coas, ys_coas, line_color="grey")
        fig.multi_line(xs_bord, ys_bord, line_color="grey")
        
        # define figure layout parameters
        fig.add_layout(legend, "above")
        fig.legend.click_policy = "hide"
        
    # check if all keyword arguments have been used.
    if len(kwargs) > 0:
        logging.warning("unknown arguments: %s" % ",".join(kwargs.keys()))
    return fig


def __individual_grid_cells(data, grid, *kwargs, regional = True, without_edges = True,
                          with_map = True,  map_resolution='110m', function = Function.TRISURF, 
                          aspectratio = 1., size = 800):
    """
    Creates an interactive 3D plot of individual grid cells.

    Parameters
    ----------
    data : xarray
        Values of a variable and a range of other parameters describing a variable and a grid.
            
    grid : xarray
        Parameters of grid cells.
        
    *kwargs
        Possibility to define a colormap.
                                          
    regional : bool, optional, default=True
        Definition of a scale. Argument equals to True for a regional scale and to False for a global scale(=whole Earth).
        
    without_edges : bool, optional, default=True
        Definition of a type of borders between individual grid cells. Argument equals to True for invisible edges between cells.
        
    with_map : bool, optional, default=True
        Possibility to add map contours above a plot.
        
    map_resolution : {'110m', '50m', '10m'}, optional, default='110m'
        Definition of a map resolution.
                 
    function : Enum element, optional, default=Function.TRISURF
        Definition of a Plotly function used for plotting. Element from Function Enum.
        
    aspectratio : float, optional, default=1.
        Ratio between a figure width and a height as a fraction. 
        
    size : int, optional, default=800
        Size of a figure width in pixels. Size of a figure height is calculated from an aspectratio.
                      
    Returns
    -------
    figure 
        Figure object contains an interactive 3D plot of individual grid cells.             
    """    
    
    # initialize a backend object
    _backend = InteractiveBackend3D()
    
    # check possibility to use given function arguments(for cells edges)
    if not without_edges:
        plot_edges = True
        if not regional:
            warnings.warn("Visualization of edges of grid cells is impossible for global scale! Cells edges are invisible at the plot.")
            without_edges = True
            plot_edges = False
        elif function == Function.MESH3D:
            warnings.warn("Visualization of edges of grid cells is impossible for MESH3D! Default function TRISURF was used instead.")
            function = Function.TRISURF
    else:
        plot_edges = False  
        
    if map_resolution not in {'110m', '50m', '10m'}:
        raise ValueError(f"unsupported map resilution: {map_resolution}! map resolution must be an instance of: '110m', '50m' or '10m'.")
        
    # define individual grid cells X- and Y-coordinates
    lons = grid.longitude_vertices.values
    lats = grid.latitude_vertices.values
    
    # define region borders
    lon0_map = np.min(np.degrees(lons)); lon1_map = np.max(np.degrees(lons))
    lat0_map = np.min(np.degrees(lats)); lat1_map = np.max(np.degrees(lats))
    
    # define approximate cell size 
    cell_size = sqrt(4*np.max(grid.cell_area_p.values)/sqrt(3))
    
    # define map paths 
    if with_map:
        xs_coas, ys_coas, zs_coas, n_coas = _backend.make_path_map3D('coastlines', cell_size, lons, lats, regional,
                                                                     lon0_map, lon1_map, lat0_map, lat1_map, map_resolution)
        xs_bord, ys_bord, zs_bord, n_bord = _backend.make_path_map3D('borders', cell_size, lons, lats, regional, 
                                                                     lon0_map, lon1_map, lat0_map, lat1_map, map_resolution) 
        
    # transform grid cells coordinates to cartesian coordinates 
    cartesian_coordinates = _backend.spherical_to_cartesian(lons, lats, radius=6371229.0)    
    xs = cartesian_coordinates.T[0]
    ys = cartesian_coordinates.T[1]
    zs = cartesian_coordinates.T[2]

    xratio, yratio, zratio = _backend.set_aspectratio3D(xs, ys, zs)
    
    # define a colormap
    colormap = 'jet'
    if len(kwargs) != 0:
        colormap = kwargs[0]
    
    # define colors of individual grid cells
    colors = _backend.set_colors(data[0,0,:].values, colormap, function) 
    
    # define figure parameters
    # define a colorbar
    colorbar = _backend.make_colorbar_continuous(data[0,0,:].values, colormap)
    
    # define layout parameters
    layout_base = {'xaxis': { 'showgrid': False, 'zeroline': False, 'visible': False},
                   'yaxis': { 'showgrid': False, 'zeroline': False, 'visible': False}}  
    scene = {'xaxis': { 'showgrid': False, 'zeroline': False, 'visible': False},
             'yaxis': { 'showgrid': False, 'zeroline': False, 'visible': False},
             'zaxis': { 'showgrid': False, 'zeroline': False, 'visible': False}}
    
    # define a legend position 
    legend = dict(yanchor="top", xanchor="right")
    
    if function == Function.TRISURF:
        
        # define individual grid cells and add them to a figure 
        simplices = grid.vertex_of_cell.T.values - 1
        fig = go.Figure(data=ff.create_trisurf(x=xs, y=ys, z=zs, color_func=colors, simplices=simplices, 
                                               plot_edges=plot_edges, aspectratio=dict(x=xratio, y=yratio, z=zratio)))
        
    elif function == Function.MESH3D:
        
        # define individual grid cells and add them to a figure 
        i = grid.vertex_of_cell.values[0] - 1
        j = grid.vertex_of_cell.values[1] - 1
        k = grid.vertex_of_cell.values[2] - 1
        fig = go.Figure(data = go.Mesh3d(x=xs, y=ys, z=zs, facecolor=colors, i=i, j=j, k=k))
        
    else:
        raise ValueError(f"unsupported function: {function}! function must be an instance of Function Enum: TRISURF or MESH3D.")
        
    if with_map:
  
        # add coastlines to a figure
        fig.add_trace(go.Scatter3d(x=xs_coas[0], y=ys_coas[0], z=zs_coas[0], mode='lines', 
                                   legendgroup='map', name='map', showlegend=True, line=dict(color='grey', width=2)))
        for i in range(1, n_coas): 
            fig.add_trace(go.Scatter3d(x=xs_coas[i], y=ys_coas[i], z=zs_coas[i], mode='lines', 
                                       legendgroup='map', name='map', showlegend=False, line=dict(color='grey', width=2)))
            
        # add country borders to a figure
        for i in range(n_bord): 
            fig.add_trace(go.Scatter3d(x=xs_bord[i], y=ys_bord[i], z=zs_bord[i], mode='lines', 
                                       legendgroup='map', name='map', showlegend=False, line=dict(color='grey', width=2)))
        
    # add a colorbar 
    fig.add_trace(colorbar)
        
    # update a layout
    fig.update_layout(layout_base)
    fig.update_layout(width=size, height=size/aspectratio, scene=scene, legend=legend, plot_bgcolor="#FFFFFF")
    
    # define a camera eye and a camera position for regional scale
    if regional:
        camera_eye = _backend.set_camera_eye(lons, lats)
        camera = dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), 
                      eye=dict(x=1.7*camera_eye[0], y=1.7*camera_eye[1], z=camera_eye[2]))
        fig.update_layout(scene_camera = camera, scene_dragmode='pan') 
    
    return fig


def __interactive_3Dcontours(data, previous_figure, *kwargs, grid = None):
    """
    Creates an interactive 3D contours on already existing plot of individual grid cells.

    Parameters
    ----------
    data : xarray
        Values of a variable and a range of other parameters describing a variable and a grid.
            
    previous_figure : figure
        Pre-created figure of individual grid cells.
        
    *kwargs
        Possibility to define a colormap of contours.
        
    grid : xarray, optional, default=None
        Parameters of grid cells for icosahedral coordinates.
                      
    Returns
    -------
    figure 
        Figure object contains an interactive 3D plot of individual grid cells with contours.             
    """    
    
    # initialize a backend object
    _backend = InteractiveBackend3D()
    
    # if coordinates are icosahedral: convert to regular-lat-lon coordinates
    if grid != None:
        data = _backend.icosahedral_to_regular_coords(grid, data)
        
    # define names of regular-lat.lon coordinates
    lon, lat = _backend.coords_name(data)
    
    # define a base figure 
    fig = previous_figure
    
    # define contours parameters
    variable_name = data._name    
    x_colorbar = 1.2
    colormap = "BuPu"
    if len(kwargs) != 0:
        colormap = kwargs[0]
    
    # define contours and its colorbar
    xs_degrees, ys_degrees, colors, values = _backend.get_contour_data(data[lon].values, data[lat].values, data[0,0,:,:].values,
                                                                       colormap, filled_contours=False)
    
    xs, ys, zs, npaths = _backend.paths_cartesian(xs_degrees, ys_degrees, radius=6401229.0)
    
    colorset, valueset, tick_interval = _backend.generate_colorscale(colors, values)
    
    colorbar = _backend.make_colorbar(colorset, valueset, tick_interval, variable_name, x_colorbar)
    
    # add contours and a colorbar to a figure
    fig.add_trace(go.Scatter3d(x=xs[0], y=ys[0], z=zs[0], mode='lines', legendgroup=variable_name, name=variable_name,
                               showlegend=True, line=dict(color=colors[0],width=4)))
    
    for i in range(1, npaths): 
        fig.add_trace(go.Scatter3d(x=xs[i], y=ys[i], z=zs[i], mode='lines', legendgroup=variable_name, name=variable_name,
                                   showlegend=False, line=dict(color=colors[i],width=4)))
        
    fig.add_trace(colorbar) 
    
    return fig


def __interactive_3Dstreamlines(data_horizontal, data_vertical, previous_figure, *kwargs, linetype = Stream.LINE, grid = None,
                              density = 2, scale = 0.02):
    """
    Creates an interactive 3D streamlines or vectors on already existing plot of individual grid cells.

    Parameters
    ----------
    datau, datav : xarray
        Values of a variable and a range of other parameters describing a variable and a grid.
        
    linetype : Enum element, optional, default=Stream.LINE
        Definition of a linetype. Element from Stream Enum.
            
    previous_figure : figure
        Pre-created figure of individual grid cells.
        
    *kwargs
        Possibility to define a color of streamlines or vectors.
        
    grid : xarray, optional, default=None
        Parameters of grid cells.
        
    density : float, optional, default=2
        Controls the closeness of streamlines. When density = 1, the domain is divided into a 30x30 grid.
                     
    scale : float, optional, default=0.02
        Possibility to scale a size of arrows(ideally to avoid overlap).
                      
    Returns
    -------
    figure 
        Figure object contains an interactive 3D plot of individual grid cells with streamlines or vectors.             
    """   
    
    # initialize a backend object
    _backend = InteractiveBackend3D()
    _backend_bokeh = InteractiveBackendBokeh()  

    # if coordinates are icosahedral: convert to regular-lat-lon coordinates
    if grid != None:
        data_horizontal = _backend.icosahedral_to_regular_coords(grid, data_horizontal)
        data_vertical   = _backend.icosahedral_to_regular_coords(grid, data_vertical)
        
    # define names of regular-lat.lon coordinates
    lon, lat = _backend.coords_name(data_horizontal)
        
    # define lines borders
    lon0 = float(np.min(data_horizontal[lon])); lon1 = float(np.max(data_horizontal[lon]))
    lat0 = float(np.min(data_horizontal[lat])); lat1 = float(np.max(data_horizontal[lat]))
    
    # define a base figure 
    fig = previous_figure
    
    # define figure parameters
    color = "black"
    if len(kwargs) != 0:
        color = kwargs[0]
    
    # define streamlines paths
    if linetype == Stream.LINE:
        
        line_name = 'streamlines'
        
        streamlines = _backend.get_streamline_data(data_horizontal[lon].values, data_horizontal[lat].values, 
                                                   data_horizontal[0,0,:,:].values, data_vertical[0,0,:,:].values, density)
        
        xs_degrees, ys_degrees = _backend_bokeh.separate_streamline_data(streamlines)
        
        # transform to cartesian coordinates and separate coordinates
        xs, ys, zs, npaths = _backend.paths_cartesian(xs_degrees, ys_degrees, radius=6401229.0)
        
    # reduce a number of grid cells for vectors and define a reference arrow
    elif linetype == Stream.VECTOR:
        
        line_name = 'vectors'
        
        # reduce a number of grid cells
        lons, lats, us, vs = _backend.reduce_vector_data(data_horizontal[lon].values, data_horizontal[lat].values,
                                                         data_horizontal[0,0,:,:].values, data_vertical[0,0,:,:].values, density)
        
        # define a position of a reference arrow and its length
        lat1_arrow = abs(int(0.1*(lat1-lat0)))
        lat1 = lat1 + lat1_arrow
        ref_arrow = _backend.get_reference_arrow_data(us, vs)
        ref_arrow_string= (str(ref_arrow) + ' ' + data_horizontal.units)
        
        # define vectors and a reference arrow
        xs_degrees, ys_degrees, xs_ref_arrow_degrees, ys_ref_arrow_degrees = _backend_bokeh.separate_vector_data(lons, lats, us, vs,
                                                                                                                 scale, ref_arrow,
                                                                                                                 lon0, lat1, 0)
        # transform to cartesian coordinates and separate coordinates
        xs, ys, zs, npaths = _backend.paths_cartesian(xs_degrees, ys_degrees, radius=6401229.0)
        xs_ref_arrow, ys_ref_arrow, zs_ref_arrow, npaths_ref_arrow = _backend.paths_cartesian(xs_ref_arrow_degrees,
                                                                                              ys_ref_arrow_degrees,
                                                                                              radius=(1+0.01*lat1_arrow)*6401229.0)       
    else:
        raise ValueError(f"unsupported lines type: {linetype}! line type must be an instance of Stream Enum: LINE or VECTOR.")
    
    # add streamlines to a figure   
    fig.add_trace(go.Scatter3d(x=xs[0], y=ys[0], z=zs[0], mode='lines', legendgroup=line_name, name=line_name,
                               showlegend=True, line=dict(color=color,width=4)))
    for i in range(1, npaths): 
        fig.add_trace(go.Scatter3d(x=xs[i], y=ys[i], z=zs[i], mode='lines', legendgroup=line_name, name=line_name,
                                   showlegend=False, line=dict(color=color,width=4))) 
    
    # add reference arrow for vectors
    if linetype == Stream.VECTOR:
        
        for i in range(npaths_ref_arrow):
            fig.add_trace(go.Scatter3d(x=xs_ref_arrow[i], y=ys_ref_arrow[i], z=zs_ref_arrow[i], mode='lines', legendgroup=line_name,
                                       name=line_name, showlegend=False, line=dict(color=color,width=4))) 
        
        fig.update_layout(scene=dict(annotations=[dict(showarrow=False, 
                                                       x=xs_ref_arrow[0][0], y=ys_ref_arrow[0][0], 
                                                       z=(1 + 0.005*lat1_arrow)*zs_ref_arrow[0][0],
                                                       text=ref_arrow_string)]))
    
    return fig

