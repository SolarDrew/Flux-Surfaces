# -*- coding: utf-8 -*-
"""
This script was used to generate the animation. It is not possible to provide 
the data used to generate this animation as it totals 191Gb in size.
All the code used to generate the data is avalible here:
https://bitbucket.org/smumford/period-paper
all be it with currently limited documentation!!
"""
from __future__ import print_function
import sys
import os
import glob

import numpy as np
import yt.mods as ytm
from mayavi import mlab
from tvtk.util.ctf import PiecewiseFunction
from tvtk.util.ctf import ColorTransferFunction
from tvtk.api import tvtk

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

mlab.options.offscreen = True
fig = mlab.figure(size=(3200, 2800))

timeseries = ytm.load(os.path.join(cfg.gdf_dir, '*5_00???.gdf'))
n = 150
ds = timeseries[n]
cg = ds.index.grids[0]
cube_slice = np.s_[:,:,:-5]

# Get displacement for a single surface and make a thing for it
linesurf = glob_files('r30','Fieldline_surface*')
surf_poly = ttf.read_step(linesurf[n])

# Get all displacement values for plotting
radii = range(6, 60, 6)
heights = range(5, 125, 10)
alldists = np.zeros((len(radii), len(heights), 187, 100))
radcoords = np.zeros((len(radii), len(heights)))
allcoords = np.zeros((3, len(radii), len(heights), 100))
for j, r in enumerate(radii):
    tube_r = 'r{:02}'.format(r)
    for i, h in enumerate(heights):
      try:
        dis = np.load(os.path.join(cfg.data_dir, tube_r, "{}_h{}_distance.npy".format(cfg.get_identifier(), h)))[:187] #[:198]
        n_t, n_theta = dis.shape
        radcoords[j, i] = dis.mean()
        if radcoords[j, i] > 64:
            allcoords[0, j, i, :] = np.nan
            allcoords[1, j, i, :] = np.nan
            allcoords[2, j, i, :] = h
            alldists[j, i, :, :] = np.nan
            continue
        d1 = dis - dis[0][None]
        #print('----', r, h, radcoords[j, i])
        d1 *= 15.6e3

        allcoords[0, j, i, :] = (radcoords[j, i] * np.cos(np.linspace(0, 2*np.pi, 100))) + 64
        allcoords[1, j, i, :] = (radcoords[j, i] * np.sin(np.linspace(0, 2*np.pi, 100))) + 64
        allcoords[2, j, i, :] = h #* 12.5e3
        alldists[j, i, :, :] = d1
      except IOError:
        print('failed radius {}, height {}'.format(r, h), file=sys.stderr)
      except IndexError:
        print('failed radius {}, height {}'.format(r, h), file=sys.stderr)

for r, rad in enumerate(radii[::2]):
    r2 = (r*2)
    for theta in range(0, 100, 25):
        dr = -alldists[r2, :, n, theta] / np.nanmax(np.abs(alldists[:, :, n, :]), axis=(0, 2))
        line = mlab.plot3d(allcoords[0, r2, :, theta], allcoords[1, r2, :, theta], allcoords[2, r2, :, theta],
                           dr, tube_radius=0.8, figure=fig, colormap='RdBu',
                           vmin=-1, vmax=1)
for h in [2, 6, 10]:
    dr = -alldists[r2, h, n, :] / np.nanmax(np.abs(alldists[:, h, n, :]))
    line = mlab.plot3d(allcoords[0, r2, h, :], allcoords[1, r2, h, :], allcoords[2, r2, h, :],
                       dr, tube_radius=0.8, figure=fig, colormap='RdBu',
                       vmin=-1, vmax=1)

#Create a bfield tvtk field, in mT
bfield = mlab.pipeline.vector_field(cg['mag_field_x'][cube_slice] * 1e3,
                                    cg['mag_field_y'][cube_slice] * 1e3,
                                    cg['mag_field_z'][cube_slice] * 1e3,
                                    name="Magnetic Field",figure=fig)
#Create a scalar field of the magntiude of the vector field
bmag = mlab.pipeline.extract_vector_norm(bfield, name="Field line Normals")

velslice = cube_slice
x = cg['x'][velslice]
y = cg['y'][velslice]
z = cg['z'][velslice]
vx = cg['velocity_x'][velslice]
vy = cg['velocity_y'][velslice]
vz = cg['velocity_z'][velslice]
vvec = mlab.pipeline.vector_field(cg['velocity_x'][cube_slice],
                                  cg['velocity_y'][cube_slice],
                                  cg['velocity_z'][cube_slice],
                                  name="Velocity Field",figure=fig)
vmag = mlab.pipeline.extract_vector_norm(vvec, name='Velocity normals')
xmax, ymax, zmax = ds.domain_dimensions - 1
#Define domain parameters
domain = {'xmax':xmax,'ymax':ymax,'zmax':10}
#Create initial seed points in tvtk.PolyData
flowseeds = ttf.make_circle_seeds(10, 10, **domain)

vsrc = mlab.pipeline.vector_scatter(vx, vy, vz)
vvecs = mlab.pipeline.vectors(vsrc, mask_points=128, scale_factor=12)

#==============================================================================
# Plotting
#==============================================================================
text_color = (1,1,1)

# Plot Surface
lim = np.nanmax(np.abs(alldists))
new_tube, surf_bar, surf_bar_label = mpf.draw_surface(surf_poly, 'RdBu', lim=lim,
                                                      position=[0.81, 0.1],
                                                      position2=[0.11,0.31])

mpf.change_surface_scalars(new_tube, surf_bar_label, 'surface_r_pos')

new_tube.parent.scalar_lut_manager.label_text_property.color = (1,1,1)
surf_bar_label.property.color = text_color
surf_bar_label.y_position = 0.05
surf_bar_label.x_position = 0.93

# Add The axes
axes, outline = mpf.add_axes(np.array(zip([-1, -1, 0], [1, 1, 1.6])).flatten(), obj=vvec)
axes.axes.property.color = text_color
axes._title_text_property.color = text_color
axes.label_text_property.color = text_color
outline.visible = False
axes.axes.y_axis_visibility = True
axes.axes.z_axis_visibility = True #False

mlab.view(150, 60, 450, 'auto')
fig.scene.save('/data/sm1ajl/Flux-Surfaces/figs/3D-bfield.png')
