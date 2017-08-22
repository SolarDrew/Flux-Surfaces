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

#t = sys.argv[1]
#fname = sys.argv[2]

ts = glob.glob('/fastdata/sm1ajl/Flux-Surfaces/gdf/m0_p120-0_0-5_0-5/m0_p120-0_0-5_0-5_00???.gdf')
ts.sort()

for t in range(100, len(ts)+1, 20):
    ds = yt.load(ts[t])
    #ds = yt.load(fname)
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
    
    """seeds = []
    for x in range(8, 256, 16):
        for y in range(8, 256, 16):
            seeds.append((x, y, 120))"""
    #B_lower = cg['magnetic_field_strength'][:, :, 10]
    #seeds = np.where(B_lower > B_lower.max()*0.5)
    #seeds = zip(seeds[0], seeds[1], [10]*len(seeds[0]))
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
    #slines.seed.widget.theta_resolution = 9
    #slines.seed.widget.radius = 40
    #slines.seed.widget.resolution = 50
    slines.seed.visible = False #Hide the seed widget
    
    vsurf = mlab.contour3d(cg['velocity_z'], colormap='RdBu')#, opacity=0.5)
    bmag.add_child(vsurf)
    #betasurf.parent.scalar_lut_manager.lut_mode = 'PiYG'
    #betasurf.parent.scalar_lut_manager.lut.scale = 'log10'
    
    # Add The axes
    axes, outline = mpf.add_axes(np.array(zip((-1, -1, 0), (1, 1, 1.6))).flatten(), obj=bfield)
    axes.axes.property.color = text_color
    axes._title_text_property.color = text_color
    axes.label_text_property.color = text_color
    outline.visible = False
    ### mlab.view(azimuth, elevation, distance, focalpoint)
    mlab.view(150, 60, 600, 'auto')
    #mlab.view(-90, 90, 1000, 'auto')
    
    fig.scene.save('/data/sm1ajl/Flux-Surfaces/figs/m0/3D/domain_{:03}.png'.format(t))
    """fig.scene.close()
    mlab.clf()
    mlab.close()
    ds.close()"""
    #del fig, bfield, bmag, axes, slines, outline, ds, cg, surf_seeds_poly, seeds, vsurf
