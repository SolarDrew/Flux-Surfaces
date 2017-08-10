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

vscal = mlab.pipeline.scalar_field(cg['velocity_magnitude'], figure=fig)

vplane1 = mlab.pipeline.scalar_cut_plane(vscal, plane_orientation='x_axes',# slice_position=0,
                                         colormap='gist_heat', figure=fig, opacity=0.6)#transparent=True)
vplane2 = mlab.pipeline.scalar_cut_plane(vscal, plane_orientation='y_axes',# slice_position=0,
                                         colormap='gist_heat', figure=fig, opacity=0.6)#transparent=True)
vplane3 = mlab.pipeline.scalar_cut_plane(vscal, plane_orientation='z_axes',# slice_position=0,
                                         colormap='gist_heat', figure=fig, opacity=0.3)#transparent=True)
vplane1.implicit_plane.origin = [128, 128, 64]
vplane2.implicit_plane.origin = [128, 128, 64]
#vplane3.implicit_plane.origin = [128, 128, 10]
vplane1.implicit_plane.widget.enabled = False
vplane2.implicit_plane.widget.enabled = False
vplane3.implicit_plane.widget.enabled = False
#import pdb; pdb.set_trace()

#==============================================================================
# Plotting
#==============================================================================
text_color = (1,1,1)
# Add The axes
axes, outline = mpf.add_axes(np.array(zip([-1, -1, 0], [1, 1, 1.6])).flatten(), obj=vscal)
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
#for n in range(860,len(timeseries),20):
for n in range(10, len(timeseries), 20):
    t1 = time.clock()
    print n
    ds = timeseries[n]
    cg = ds.index.grids[0]
    current_time = ds.current_time

    vlim = abs(cg['velocity_magnitude']).max()
    print timeseries[n], cg['velocity_z'].min(), cg['velocity_z'].max(), vlim

    vscal.set(scalar_data=cg['velocity_magnitude'])#, vmin=-100, vmax=100)
    vscal.mlab_source.set(vmin=-vlim, vmax=vlim)

    idx = np.unravel_index(cg['velocity_magnitude'].argmax(), cg['velocity_magnitude'].shape)
    vplane3.implicit_plane.widget.enabled = True
    vplane3.implicit_plane.origin = [128, 128, idx[2]]
    vplane3.implicit_plane.widget.enabled = False

    vplane1.update_pipeline()
    vplane2.update_pipeline()
    vplane3.update_pipeline()

    mlab.view(150, 60, 600, 'auto')
    fig.scene.save('/data/sm1ajl/Flux-Surfaces/figs/m0/3D/vmag/domain_{:03}.png'.format(n))
    print "step %i done in %f s\n"%(n,time.clock()-t1)+"_"*80+"\n"

t2 = time.clock()
print "All done in %f s %f min\n"%(t2-t,t2-t/60.)+"_"*80+"\n"
