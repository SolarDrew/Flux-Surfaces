from __future__ import print_function
import sys
import os
import glob

import numpy as np
from matplotlib import use
use('agg')
import matplotlib.pyplot as plt
import matplotlib.colors as col
from yt.funcs import mylog
import yt.mods as ytm
from mayavi import mlab
from tvtk.util.ctf import PiecewiseFunction
from tvtk.util.ctf import ColorTransferFunction
from tvtk.api import tvtk
import tvtk.common as tvtkc

from astropy.io import fits

#pysac imports
#import pysac.io.yt_fields
import pysac.yt
import pysac.analysis.tube3D.tvtk_tube_functions as ttf
import pysac.plot.mayavi_plotting_functions as mpf

# Import this repos config
sys.path.append("../")
from scripts.sacconfig import SACConfig
m = sys.argv[1]
n = int(sys.argv[2])
tube_r = 'r{:02}'.format(int(sys.argv[3]))
os.system('./configure.py set SAC --usr_script=m{}'.format(m))
cfg = SACConfig()

mylog.setLevel(40)
cube_slice = np.s_[:,:,:-5]

saveout = True


def mlab_view(scene, obj, azimuth=153, elevation=62, distance=400,
              focalpoint=np.array([25., 63., 60.]), aa=16):
    scene.anti_aliasing_frames = aa
    mlab.view(azimuth=azimuth, elevation=elevation, distance=distance, focalpoint=focalpoint)

    # Add The axes
    text_color = (0, 0, 0)
    axes, outline = mpf.add_axes(np.array(zip([-1, -1, 0], [1, 1, 1.6])).flatten(), obj=obj)
    axes.axes.property.color = text_color
    axes._title_text_property.color = text_color
    axes.label_text_property.color = text_color
    outline.visible = False
    axes.axes.y_axis_visibility = True
    axes.axes.z_axis_visibility = True #False

    fname = '/data/sm1ajl/Flux-Surfaces/figs/poster/surface_dr-slice_m{}_t{}.png'.format(m, n)
    scene.save(fname, size=(3200, 2800))


def glob_files(tube_r, search):
    files = glob.glob(os.path.join(cfg.data_dir,tube_r,search))
    files.sort()
    return files


def get_rpos(ds, tube_r):
    cg = ds.index.grids[0]
    # Create a bfield tvtk field, in mT
    bfield = mlab.pipeline.vector_field(cg['mag_field_x'][cube_slice] * 1e3,
                                        cg['mag_field_y'][cube_slice] * 1e3,
                                        cg['mag_field_z'][cube_slice] * 1e3,
                                        name='Magnetic Field', figure=None)
    ## Create a scalar field of the magnitude of the vector field
    bmag = mlab.pipeline.extract_vector_norm(bfield, name='Field line Normals')
    
    # Define the size of the domain
    xmax, ymax, zmax = np.array(cg['mag_field_x'][cube_slice].shape)# - 1
    domain = {'xmax': xmax, 'ymax': ymax, 'zmax': zmax}
    
    """if tube_r == 'r12':
        angles = None
    else:
        angles=np.linspace(0, 2*np.pi, 100, endpoint=False)[:60] + (0.58 * np.pi)"""
    angles = None
    surf_seeds_poly = ttf.make_circle_seeds(100, int(tube_r[1:]), angles=angles, **domain)
    
    #Make a streamline instance with the bfield
    surf_field_lines = tvtk.StreamTracer()
    #bfield is a mayavi data object, we require a tvtk dataset which can be access thus:
    tvtkc.configure_input(surf_field_lines, bfield.outputs[0])
    
    tvtkc.configure_source_data(surf_field_lines, surf_seeds_poly)
    surf_field_lines.integrator = tvtk.RungeKutta4()
    surf_field_lines.maximum_propagation = 1000
    surf_field_lines.integration_direction = 'backward'
    surf_field_lines.update()
    
    #Create surface from 'parallel' lines
    surface = tvtk.RuledSurfaceFilter()
    tvtkc.configure_input(surface, surf_field_lines.output)
    if tube_r == 'r12':
        surface.close_surface = True
    else:
        surface.close_surface = False
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

    return distance, surface
    

def plot_rshift(shift, surface):
    pd_dis = tvtk.PointData(scalars=shift)
    pd_dis.scalars.name = "r_shift"
    
    poly_out = surface.output
    poly_out.point_data.add_array(pd_dis.scalars)
    
    flux_surface2 = mlab.pipeline.surface(surface.output, vmin=0, vmax=1)
    
    #Set the surface component to be the radial displacement
    flux_surface2.parent.parent.point_scalars_name = 'r_shift'
    
    cmap = plt.get_cmap('coolwarm')(range(255))*255
    flux_surface2.module_manager.scalar_lut_manager.lut.table = cmap 
    

mlab.options.offscreen = True
fig = mlab.figure(bgcolor=(1,1,1))
scene = fig.scene

timeseries = ytm.load(os.path.join(cfg.gdf_dir, '*5_00???.gdf'))
#n = 150
"""rpos0 = [None, None, None]
for a, tube_r in enumerate(['r12', 'r33', 'r63']):
    rpos0[a], surface = get_rpos(timeseries[0], tube_r)
"""
for n in [n]: #range(10, 180, 10):
    ds = timeseries[n]
    cg = ds.index.grids[0]
    
    try:
        #for a, tube_r in enumerate(['r12', 'r33', 'r63']):
        #for a, rad in enumerate(range(3, 66, 3)):
            a = 0
            #tube_r = 'r{:02}'.format(rad)
            rpos0, surface = get_rpos(timeseries[0], tube_r)
            rpos1, surface = get_rpos(ds, tube_r)
            dr = (rpos1 - rpos0)*15.6
            if a == 0:
                lim = max(abs(dr.min()), dr.max())
            print(dr.min(), dr.max(), lim)
            if saveout:
                data_dir = os.path.join(cfg.data_dir,'%s/'%tube_r)
                outname = os.path.join(data_dir, 'surface_displacement_{}_{}_{:05d}.vtp'.format(tube_r, cfg.get_identifier(), n+1))
                writer = ttf.PolyDataWriter(outname, surface.output)
                writer.add_array(perp=normals,
                                 par=parallels,
                                 phi=torsionals,
                                 vperp=vperp,
                                 vpar=vpar,
                                 vphi=vphi,
                                 bpertperp=bpertperp,
                                 bpertpar=bpertpar,
                                 bpertphi=bpertphi,
                                 surface_density=surface_density,
                                 surface_va=surface_va,
                                 surface_beta=surface_beta,
                                 surface_cs=surface_cs,
                                 surface_r_pos=surface_r_pos,
                                 surface_dr=dr)
                #writer.add_array(surface_dr=dr)
                writer.write()
                continue
            scaled_dr = col.SymLogNorm(linthresh=2e-4, vmin=-lim, vmax=lim)(value=dr)
            #scaled_dr = col.SymLogNorm(linthresh=5e-2)(value=dr)
            print(scaled_dr.min(), scaled_dr.max())
            plot_rshift(scaled_dr, surface)
    except ValueError as e:
        print('Failed at time-step {}'.format(n))
        print(e)
        continue
    if saveout:
        continue
    vx = cg['velocity_x'][cube_slice]
    vy = cg['velocity_y'][cube_slice]
    vz = cg['velocity_z'][cube_slice]
    
    vsrc = mlab.pipeline.vector_scatter(vx, vy, vz)
    vvecs = mlab.pipeline.vectors(vsrc, mask_points=128, scale_factor=10, colormap='inferno')
    vvecs.parent.vector_lut_manager.reverse_lut = True
    
    mlab_view(fig.scene, vvecs, elevation=60, azimuth=35, distance=450, focalpoint='auto')
