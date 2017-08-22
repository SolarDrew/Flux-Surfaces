from mayavi import mlab
import pysac.yt
import yt
import matplotlib.pyplot as plt
import astropy.units as u
import sys
import os
import glob

import numpy as np
import yt.mods as ytm
from tvtk.util.ctf import PiecewiseFunction
from tvtk.util.ctf import ColorTransferFunction

from astropy.io import fits

#pysac imports
import pysac.io.yt_fields
import pysac.analysis.tube3D.tvtk_tube_functions as ttf
import pysac.plot.mayavi_plotting_functions as mpf
import pysac.plot.mayavi_seed_streamlines as mss

#ds = yt.load('/fastdata/sm1ajl/driver-widths/gdf/Slog_p90-0_4-8_0-30/Slog_p90-0_4-8_0-30_00001.gdf')
ds = yt.load('/data/sm1ajl/custom_ini_files/paper2b/paper2b.gdf')

cg = ds.index.grids[0]
cube_slice = np.s_[:,:,:]#-5]
print(cg.shape)
text_color = (1, 1, 1)

mlab.options.offscreen = True
fig = mlab.figure(size=(800, 700))

#Create a bfield tvtk field, in mT
bfield = mlab.pipeline.vector_field(cg['mag_field_x'][cube_slice] * 1e3,
                                    cg['mag_field_y'][cube_slice] * 1e3,
                                    cg['mag_field_z'][cube_slice] * 1e3,
                                    name="Magnetic Field",figure=fig)
#Create a scalar field of the magntiude of the vector field
bmag = mlab.pipeline.extract_vector_norm(bfield, name="Field line Normals")

"""seeds = []
for x in range(0, 51, 10):
    for y in range(0, 51, 10):
        seeds.append((x, y, 10))#420))"""
B_lower = cg['magnetic_field_strength'][:, :, 10]
seeds = np.where(B_lower > B_lower.max()*0.45)
seeds = zip(seeds[0], seeds[1], [10]*len(seeds[0]))
slines = mss.SeedStreamline(seed_points=seeds[::4])
bmag.add_child(slines)
slines = mlab.pipeline.streamline(bmag, linetype='tube',
                                  integration_direction='both', seed_resolution=6)
slines.stream_tracer.maximum_propagation = 250#500 #Make sure the lines hit the edge of the domain
slines.tube_filter.radius = 0.3
slines.parent.scalar_lut_manager.lut_mode = 'GnBu'
slines.parent.scalar_lut_manager.lut.scale = 'log10'
#slines.seed.widget.theta_resolution = 9
#slines.seed.widget.radius = 40
#slines.seed.widget.resolution = 50
slines.seed.visible = False #Hide the seed widget
# Tweak to make the lower limit not zero for log scaling
##slines.parent.scalar_lut_manager.data_range = np.array([1e-5,1e-2])
# Add colour bar
"""cbar = mpf.add_colourbar(slines, [0.81, 0.5] ,[0.11,0.31], '', label_fstring='%#3.1e',
                  number_labels=5, orientation=1,lut_manager='scalar')
cbar_label = mpf.add_cbar_label(cbar,'Magnetic Field Strength\n               [mT] ')
cbar_label.property.color = text_color
slines.parent.scalar_lut_manager.label_text_property.color = (1,1,1)
cbar_label.y_position = 0.45
cbar_label.x_position = 0.93"""

betasurf = mlab.contour3d(np.log10(cg['plasma_beta']), colormap='PiYG', opacity=0.75, contours=[a for a in range(-1, 6)])
bmag.add_child(betasurf)
#betasurf.parent.scalar_lut_manager.lut_mode = 'PiYG'
#betasurf.parent.scalar_lut_manager.lut.scale = 'log10'

# Add The axes
#axes, outline = mpf.add_axes(np.array(zip(ds.domain_left_edge,ds.domain_right_edge)).flatten()/1e8, obj=bfield)
#axes, outline = mpf.add_axes(np.array(zip((0, 0, 0), (2.0, 2.0, 1.6))).flatten(), obj=bfield)
#axes, outline = mpf.add_axes(np.array(zip((-1.6, -0.8, 0), (1.6, 0.8, 8.62))).flatten(), obj=bfield)
axes, outline = mpf.add_axes(np.array(zip((-0.5, -0.5, 0), (0.5, 0.5, 2.78))).flatten(), obj=bfield)
axes.axes.property.color = text_color
axes._title_text_property.color = text_color
axes.label_text_property.color = text_color
outline.visible = False
#axes.axes.y_axis_visibility = True
#axes.axes.z_axis_visibility = False
#mlab.view(-90.0, 75.0, 380.0, [ 70.0,  56.4,  61.5])
#mlab.view(30, 67, 400, 'auto')
### mlab.view(azimuth, elevation, distance, focalpoint)
mlab.view(150, 90, 400, 'auto')
#mlab.view(-90, 90, 1000, 'auto')

fig.scene.save('/home/sm1ajl/Flux-Surfaces/ini_conditions_3D.png')
