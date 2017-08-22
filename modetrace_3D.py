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
xmax, ymax, zmax = ds.domain_dimensions - 1
#Define domain parameters
domain = {'xmax':xmax,'ymax':ymax,'zmax':10}
#Create initial seed points in tvtk.PolyData
flowseeds = ttf.make_circle_seeds(10, 30, **domain)

points = [flowseeds.points]

#==============================================================================
# Plotting
#==============================================================================
text_color = (1,1,1)

# Add The axes
axes, outline = mpf.add_axes(np.array(zip([-1, -1, 0], [1, 1, 1.6])).flatten(), obj=vvec)
axes.axes.property.color = text_color
axes._title_text_property.color = text_color
axes.label_text_property.color = text_color
outline.visible = False
axes.axes.y_axis_visibility = True
axes.axes.z_axis_visibility = True #False

# Now let's try and animate this
print len(timeseries)
import time
t = time.clock()
times = []
previous_time = 0
for n in range(20,120,20):
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

    vvec.set(vector_data = np.rollaxis(np.array([cg['velocity_x'][cube_slice],
                                                 cg['velocity_y'][cube_slice],
                                                 cg['velocity_z'][cube_slice]]),
                                                 0, 4))

    newseeds = ttf.move_seeds(flowseeds, vvec.outputs[0], current_time-previous_time)
    points.append(flowseeds.points)
    previous_time = ds.current_time
    times.append(ds.current_time)

    print "step %i done in %f s\n"%(n,time.clock()-t1)+"_"*80+"\n"

t2 = time.clock()
print "All done in %f s %f min\n"%(t2-t,t2-t/60.)+"_"*80+"\n"

import pdb; pdb.set_trace()

#Tweak the figure and set the view
mlab.view(150, 60, 600, 'auto')
#fig.scene.anti_aliasing_frames = 1
fig.scene.save('/data/sm1ajl/Flux-Surfaces/figs/m0/3D/particle_trace.png')
