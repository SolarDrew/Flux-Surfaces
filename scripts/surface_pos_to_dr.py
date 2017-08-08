import sys
import os
import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt
import yt
from tvtk.api import tvtk

import pysac.yt
import pysac.analysis.tube3D.tvtk_tube_functions as ttf
import pysac.plot.mayavi_plotting_functions as mpf
from pysac.plot.mayavi_seed_streamlines import SeedStreamline

sys.path.append('../')
from scripts.sacconfig import SACConfig
cfg = SACConfig()

from IPython.core.display import Image

def mlab_view(scene, t, azimuth=153, elevation=62, distance=400,
              focalpoint=np.array([25., 63., 60.]), aa=16):
    scene.anti_aliasing_frames = aa
    mlab.view(azimuth=azimuth, elevation=elevation, distance=distance, focalpoint=focalpoint)
    fname = '/data/sm1ajl/driver-widths-paper/figs/{}/3D_tube_view/{}_{:03}.png'.format(
        str(cfg.delta_x).replace('.', '-'), cfg.get_identifier(), t)
    scene.save(fname, size=(600, 600))
    mlab.clf()
    return Image(filename=fname)

#timeseries = yt.load('/fastdata/sm1ajl/driver-widths/gdf/Slog_p90-0_10-_0-15/Slog_p90-0_10-_0-15_00???.gdf')
timeseries = yt.load('/fastdata/sm1ajl/driver-widths/gdf/Slog_p{0}_{1}_{2}/Slog_p{0}_{1}_{2}_00???'.format(
                        cfg.period, cfg.amp[:3], cfg.delta_x).replace('.', '-')+'.gdf')

# Define which step
#n = 300

# Slices
cube_slice = np.s_[:,:,:-5]
x_slice = np.s_[:,:,:,:-5]

tube_r = cfg.tube_radii[2]

# if running this creates a persistant window just get it out of the way
mlab.options.offscreen = True
fig = mlab.figure(bgcolor=(1,1,1))
scene = fig.scene

# Use first timestamp
#ds = timeseries

def get_rpos(ds):
    cg = ds.index.grids[0]
    
    # Create a bfield tvtk field, in mT
    bfield = mlab.pipeline.vector_field(cg['mag_field_x'][cube_slice] * 1e3,
                                        cg['mag_field_y'][cube_slice] * 1e3,
                                        cg['mag_field_z'][cube_slice] * 1e3,
                                        name='Magnetic Field', figure=None)
    # Create a scalar field of the magnitude of the vector field
    bmag = mlab.pipeline.extract_vector_norm(bfield, name='Field line Normals')
    
    # Define the size of the domain
    xmax, ymax, zmax = np.array(cg['mag_field_x'][cube_slice].shape)# - 1
    domain = {'xmax': xmax, 'ymax': ymax, 'zmax': zmax}
    print domain
    
    # Add axes
    axes, outline = mpf.add_axes(np.array(zip(ds.domain_left_edge, ds.domain_right_edge)).flatten()/1e8)
    surf_seeds_poly = ttf.make_circle_seeds(100, int(tube_r[1:]), **domain)
    seeds = np.array(surf_seeds_poly.points)
    
    #Make a streamline instance with the bfield
    surf_field_lines = tvtk.StreamTracer()
    #bfield is a mayavi data object, we require a tvtk dataset which can be access thus:
    surf_field_lines.input = bfield.outputs[0]
    
    surf_field_lines.source = surf_seeds_poly
    surf_field_lines.integrator = tvtk.RungeKutta4()
    surf_field_lines.maximum_propagation = 1000
    surf_field_lines.integration_direction = 'backward'
    surf_field_lines.update()
    
    #Create surface from 'parallel' lines
    surface = tvtk.RuledSurfaceFilter()
    surface.input = surf_field_lines.output
    surface.close_surface = True
    surface.pass_lines = True
    surface.offset = 0
    surface.distance_factor = 30
    surface.ruled_mode = 'point_walk'
    surface.update()
    
    #Set the lines to None to remove the input lines from the output
    surface.output.lines = None
    
    points = np.array(surface.output.points)
    points -= [63, 63, 0]
    #print points
    
    distance = np.sqrt((points[:,0]**2 + points[:,1]**2))
    #print distance
    #print distance.shape

    return distance, surface
    

def plot_rshift(shift, surface, t):
    pd_dis = tvtk.PointData(scalars=shift)
    pd_dis.scalars.name = "r_shift"
    
    poly_out = surface.output
    poly_out.point_data.add_array(pd_dis.scalars)
    
    flux_surface2 = mlab.pipeline.surface(surface.output)
    
    #Set the surface component to be the azimuthal component
    flux_surface2.parent.parent.point_scalars_name = 'r_shift'
    
    #flux_surface2.module_manager.scalar_lut_manager.lut.table = plt.get_cmap('Reds')(range(255))*255
    flux_surface2.module_manager.scalar_lut_manager.lut.table = plt.get_cmap('coolwarm')(range(255))*255
    #flux_surface2.module_manager.scalar_lut_manager.lut.table = plt.get_cmap('Reds')(range(255))*255
    lim = np.max([np.nanmax(surface.output.point_data.scalars),
                  np.abs(np.nanmin(surface.output.point_data.scalars))])*0.5
    flux_surface2.module_manager.scalar_lut_manager.data_range = np.array([-lim,lim])
    
    surf_bar = mpf.add_colourbar(flux_surface2, [0.84, 0.2], [0.11, 0.31], title='', label_fstring='%#4.2f',
                                number_labels=5, orientation=1, lut_manager='scalar')
    
    mpf.add_cbar_label(surf_bar, 'Flux surface displacement\n         [grid points]')
    #mlab.show()
    mlab_view(fig.scene, t, elevation=90, azimuth=40, focalpoint='auto')
    #mlab_view(fig.scene, t, elevation=0, azimuth=0, focalpoint='auto')


rpos0, surface = get_rpos(timeseries[0])
print(len(timeseries))
for t in range(len(timeseries),):
    if t % 5 == 0:
        continue
    ds = timeseries[t]
    rpos1, surface = get_rpos(ds)
    #fname = '/fastdata/sm1ajl/driver-widths/data/{0}/r60/surface_position_{0}_00{1:03}'.format(
    #    cfg.get_identifier(), t) + '.vtp'
    #writer = ttf.PolyDataWriter(fname, surface.output)
    #writer.write()
    print t, rpos1.shape,
    try:
        plot_rshift(rpos1-rpos0, surface, t)
        #rpos0 = rpos1.copy()
        print
    except ValueError:
        print 'Failed'
