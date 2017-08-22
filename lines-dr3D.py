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

from sacconfig import SACConfig

cfg = SACConfig()

radii = range(3, 66, 3)
heights = range(15, 125, 30)
heights2 = [15, 65, 115]

displacements = np.zeros((len(radii)*2, len(heights), 270, 100))
surfcoords = np.zeros((len(radii)*2, len(heights)))

for j, r in enumerate(radii):
    tube_r = 'r{:02}'.format(r)
    for i, h in enumerate(heights):
        dis = np.load(os.path.join(cfg.data_dir, tube_r, "{}_h{}_distance.npy".format(cfg.get_identifier(), h)))[:270]
        n_t, n_theta = dis.shape
        radcoords[j*2, i] = dis.mean()
        d1 = dis - dis[0][None]
        d1 *= 15.6e3

        alldists[j*2, i, :, :] = d1
print('Files loaded')

t = sys.argv[1]

ts = glob.glob('/fastdata/sm1ajl/Flux-Surfaces/gdf/m0_p60-0_0-5_0-5/m0_p60-0_0-5_0-5_00???.gdf')
ts.sort()

#for t in range(100, len(ts)+1, 20):
ds = yt.load(ts[t])
cg = ds.index.grids[0]
cube_slice = np.s_[:,:,:]#-5]
print(t, cg.shape)
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

xmax, ymax, zmax = ds.domain_dimensions - 1
domain = {'xmax':xmax,'ymax':ymax,'zmax':0}
surf_seeds_poly = ttf.make_circle_seeds(25, 10, **domain)
seeds = np.array(surf_seeds_poly.points)
slines = mss.SeedStreamline(seed_points=seeds)
bmag.add_child(slines)
slines = mlab.pipeline.streamline(bmag, linetype='tube',
			      integration_direction='both', seed_resolution=6)
slines.stream_tracer.maximum_propagation = 250#500 #Make sure the lines hit the edge of the domain
slines.tube_filter.radius = 0.3
slines.parent.scalar_lut_manager.lut_mode = 'GnBu'
slines.parent.scalar_lut_manager.lut.scale = 'log10'
slines.seed.visible = False #Hide the seed widget

vsurf = mlab.contour3d(cg['velocity_z'], colormap='RdBu')#, opacity=0.5)
bmag.add_child(vsurf)

# Add The axes
axes, outline = mpf.add_axes(np.array(zip((-1, -1, 0), (1, 1, 1.6))).flatten(), obj=bfield)
axes.axes.property.color = text_color
axes._title_text_property.color = text_color
axes.label_text_property.color = text_color
outline.visible = False
### mlab.view(azimuth, elevation, distance, focalpoint)
mlab.view(150, 60, 600, 'auto')

fig.scene.save('/data/sm1ajl/Flux-Surfaces/figs/m0/3D/domain_{:03}.png'.format(t))
