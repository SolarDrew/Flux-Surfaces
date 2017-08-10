# -*- coding: utf-8 -*-
"""
This script was used to generate the animation. It is not possible to provide 
the data used to generate this animation as it totals 191Gb in size.
All the code used to generate the data is avalible here:
https://bitbucket.org/smumford/period-paper
all be it with currently limited documentation!!
"""

import sys
import os
import glob

import numpy as np
import yt.mods as ytm
from mayavi import mlab
from tvtk.util.ctf import PiecewiseFunction
from tvtk.util.ctf import ColorTransferFunction

from astropy.io import fits

#pysac imports
#import pysac.io.yt_fields
import pysac.yt
import pysac.analysis.tube3D.tvtk_tube_functions as ttf
import pysac.plot.mayavi_plotting_functions as mpf

#Import this repos config
sys.path.append("../")
from scripts.sacconfig import SACConfig
cfg = SACConfig()

def glob_files(tube_r, search):
    files = glob.glob(os.path.join(cfg.data_dir,tube_r,search))
    files.sort()
    return files

n = 400
timeseries = ytm.load(os.path.join(cfg.gdf_dir, '*5_00???.gdf'))
ds = timeseries[20]
cg = ds.index.grids[0]
cube_slice = np.s_[:,:,:-5]

mlab.options.offscreen = True
fig = mlab.figure(size=(800, 700))

#Create a bfield tvtk field, in mT
"""bfield = mlab.pipeline.vector_field(cg['mag_field_x'][cube_slice] * 1e3,
                                    cg['mag_field_y'][cube_slice] * 1e3,
                                    cg['mag_field_z'][cube_slice] * 1e3,
                                    name="Magnetic Field",figure=fig)
#Create a scalar field of the magntiude of the vector field
bmag = mlab.pipeline.extract_vector_norm(bfield, name="Field line Normals")"""

vvec = mlab.pipeline.vector_field(cg['velocity_x'][cube_slice],
                                  cg['velocity_y'][cube_slice],
                                  cg['velocity_z'][cube_slice],
                                  name="Velocity Field",figure=fig)
vmag = mlab.pipeline.extract_vector_norm(vvec, name='Velocity normals')
xmax, ymax, zmax = ds.domain_dimensions - 1
#Define domain parameters
domain = {'xmax':xmax,'ymax':ymax,'zmax':10}
#Create initial seed points in tvtk.PolyData
flowseeds = ttf.make_circle_seeds(10, 30, **domain)

vflow = mlab.pipeline.streamline(vmag, linetype='line', integration_direction='both', seedtype='plane',
                                 colormap='YlGnBu', seed_visible=False)
vflow.seed.poly_data.points = flowseeds.points
points = [flowseeds.points]
#vplane1 = mlab.pipeline.vector_cut_plane(vmag, resolution=128, colormap='YlGnBu', scale_factor=16)
#vplane2 = mlab.pipeline.vector_cut_plane(vmag, resolution=128, colormap='YlGnBu', scale_factor=16)#,
#                                         plane_orientation='y_axes')
#import pdb; pdb.set_trace()
#vflow.seed.widget.normal = [0.0, 0.0, 1.0]
#vflow.seed.widget.center = [128.5, 128.5, 10]

#lim = 36000
"""vscal = mlab.pipeline.scalar_field(cg['velocity_magnitude'], figure=fig)
vx = mlab.pipeline.scalar_field(cg['velocity_x'], figure=fig)
vy = mlab.pipeline.scalar_field(cg['velocity_y'], figure=fig)
vz = mlab.pipeline.scalar_field(cg['velocity_z'], figure=fig)"""
#vsurf = mlab.pipeline.iso_surface(vscal, colormap='RdBu', opacity=0.5,
#                                  vmin=-lim, vmax=lim, contours=5)

"""vplane1 = mlab.pipeline.scalar_cut_plane(vscal, plane_orientation='x_axes',# slice_position=0,
                                         colormap='YlGnBu', figure=fig, opacity=0.5)#transparent=True)
vplane2 = mlab.pipeline.scalar_cut_plane(vscal, plane_orientation='y_axes',# slice_position=0,
                                         colormap='YlGnBu', figure=fig, opacity=0.5)#transparent=True)
vplane3 = mlab.pipeline.scalar_cut_plane(vscal, plane_orientation='z_axes',# slice_position=0,
                                         colormap='YlGnBu', figure=fig, opacity=0.5)#transparent=True)
vplane1.implicit_plane.origin = [128, 128, 64]
vplane2.implicit_plane.origin = [128, 128, 64]
vplane3.implicit_plane.origin = [128, 128, 10]
vplane1.implicit_plane.widget.enabled = False
vplane2.implicit_plane.widget.enabled = False
vplane3.implicit_plane.widget.enabled = False"""
#vplane.ipw.slice_index = 64

#xpts, ypts, zpts = flowpoints[..., 0], flowpoints[..., 1], flowpoints[..., 2]
#import pdb; pdb.set_trace()
#trace = mlab.pipeline.scalar_scatter(xpts, ypts, zpts)
#trace = mlab.points3d(xpts, ypts, zpts, figure=fig, scale_factor=0.8)

#==============================================================================
# Plotting
#==============================================================================
text_color = (1,1,1)
# Magnetic field lines
"""slines = mlab.pipeline.streamline(bmag, linetype='tube',
                                  integration_direction='both', seed_resolution=6)
slines.stream_tracer.maximum_propagation = 500 #Make sure the lines hit the edge of the domain
slines.tube_filter.radius = 0.3
slines.parent.scalar_lut_manager.lut_mode = 'GnBu'
slines.parent.scalar_lut_manager.lut.scale = 'log10'
slines.seed.widget.theta_resolution = 9
slines.seed.widget.radius = 40
slines.seed.visible = False #Hide the seed widget"""
# Tweak to make the lower limit not zero for log scaling
#slines.parent.scalar_lut_manager.data_range = np.array([1e-5,1e-2])

# Add The axes
axes, outline = mpf.add_axes(np.array(zip([-1, -1, 0], [1, 1, 1.6])).flatten(), obj=vvec)
axes.axes.property.color = text_color
axes._title_text_property.color = text_color
axes.label_text_property.color = text_color
outline.visible = False
axes.axes.y_axis_visibility = True
axes.axes.z_axis_visibility = True #False

#Tweak the figure and set the view
mlab.view(150, 60, 600, 'auto')
#fig.scene.anti_aliasing_frames = 1

# Now let's try and animate this
print len(timeseries)
import time
t = time.clock()
times = []
previous_time = 0
for n in range(20,len(timeseries),20):
    t1 = time.clock()
    print n
    ds = timeseries[n]
    cg = ds.index.grids[0]
    current_time = ds.current_time

    vlim = abs(cg['velocity_magnitude']).max()
    print timeseries[n], cg['velocity_z'].min(), cg['velocity_z'].max(), vlim

    """bfield.set(vector_data = np.rollaxis(np.array([cg['mag_field_x'][cube_slice] * 1e3,
                                                   cg['mag_field_y'][cube_slice] * 1e3,
                                                   cg['mag_field_z'][cube_slice] * 1e3]),
                                                   0, 4))
    # Tweak to make the lower limit not zero for log scaling
    #slines.parent.scalar_lut_manager.data_range = np.array([1e-5,1e-2])
    slines.update_pipeline()"""

    #vx.set(scalar_data=cg['velocity_x'])#, vmin=-100, vmax=100)
    #vy.set(scalar_data=cg['velocity_y'])#, vmin=-100, vmax=100)
    #vz.set(scalar_data=cg['velocity_z'])#, vmin=-100, vmax=100)
    #vscal.set(scalar_data=cg['velocity_magnitude'])#, vmin=-100, vmax=100)
    #vscal.mlab_source.set(vmin=-vlim, vmax=vlim)
    vvec.set(vector_data = np.rollaxis(np.array([cg['velocity_x'][cube_slice],
                                                 cg['velocity_y'][cube_slice],
                                                 cg['velocity_z'][cube_slice]]),
                                                 0, 4))

    #vsurf.mlab_source.scalars = cg['velocity_z']
    #vsurf.update_pipeline()
    vflow.update_pipeline()
    #vplane.mlab_source.scalars = cg['velocity_z']
    """vplane1.update_pipeline()
    vplane2.update_pipeline()
    vplane3.update_pipeline()"""
    #print np.array(flowseeds.points)
    newseeds = ttf.move_seeds(flowseeds, vvec.outputs[0], current_time-previous_time)
    #print np.array(flowseeds.points)
    points.append(flowseeds.points)
    #import pdb; pdb.set_trace()
    #flowpoints = np.vstack(flowpoints, newseeds)
    #trace.set(points=newseeds)#[..., 0], newseeds[..., 1], newseeds[..., 2]
    #trace.mlab_source.points = newseeds
    #trace.update_pipeline()
    previous_time = ds.current_time
    times.append(ds.current_time)

    mlab.view(150, 60, 600, 'auto')
    fig.scene.save('/data/sm1ajl/Flux-Surfaces/figs/m0/3D/particle_trace/domain_{:03}.png'.format(n))
    print "step %i done in %f s\n"%(n,time.clock()-t1)+"_"*80+"\n"

t2 = time.clock()
print "All done in %f s %f min\n"%(t2-t,t2-t/60.)+"_"*80+"\n"
