# coding: utf-8
import pysac.yt
#import pysac.analysis.tube3D.process_utils as utils
import yt
import yt.units as u
from yt.visualization.api import get_multi_plot
import numpy as np
import matplotlib as mpl
import matplotlib.animation as ani
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axisartist import floating_axes
from matplotlib.transforms import Affine2D
import sys

"""def _v_theta(field, data):
    return u.yt_array.YTArray(np.arctan2(data['velocity_y'], data['velocity_x']), data['velocity_x'].units)


def _v_r(field, data):
    return u.yt_array.YTArray(np.sqrt(data['velocity_x']**2 + data['velocity_y']**2),
                              data['velocity_x'].units)"""


def _theta(field, data):
    x, y = data['x'], data['y']
    return u.yt_array.YTArray(np.arctan2(y, x), 'radian')


def _v_theta(field, data):
    x, y, theta = data['velocity_x'], data['velocity_y'], data['theta']
    return u.yt_array.YTArray(x*np.sin(theta) + y*np.cos(theta), x.units)


def _v_theta_abs(field, data):
    return u.yt_array.YTArray(abs(data['v_theta']), data['v_theta'].units)


def _v_r(field, data):
    x, y, theta = data['velocity_x'], data['velocity_y'], data['theta']
    return u.yt_array.YTArray(x*np.cos(theta) - y*np.sin(theta), x.units)


yt.add_field('theta', function=_theta, units='radian')
yt.add_field('v_theta', function=_v_theta, units='km/s')
yt.add_field('v_theta_abs', function=_v_theta_abs, units='km/s')
yt.add_field('v_r', function=_v_r, units='km/s')

plotvar = 'density'#_pert'

for mode in ['m0']:#, 'm1', 'm2']:
  ts = yt.load('/fastdata/sm1ajl/Flux-Surfaces/gdf/{}_p120-0_0-5_0-5/*.gdf'.format(mode))
  print len(ts)

  for t in [0, 2, 4]:#range(0, len(ts), 2):
    fig = plt.figure()#figsize=(15, 8))
    grid = AxesGrid(fig, (0.075,0.075,0.85,0.85), nrows_ncols=(1, 2),
                    cbar_location="right", cbar_mode="edge", cbar_size="5%", cbar_pad="0%")

    ds = ts[t]
    minvar, maxvar = ds.all_data().quantities.extrema(plotvar)
    extreme = max(abs(minvar), abs(maxvar))
    minv, maxv = ds.all_data().quantities.extrema("velocity_x")
    extremev = max(abs(minv), abs(maxv))

    """disk = ds.disk([1.0, 1.0, 0.1], [0.0, 0.0, 1.0], 2*u.Mm, 2*u.Mm)
    plot = yt.ProfilePlot(disk, "radius", ["mag_field_x", "mag_field_y", "mag_field_z"])
    plot.set_unit('radius', 'Mm')
    plot.save('figs/{}/magfield/{:03}'.format(mode, t))"""

    print "====", minv, maxv, '===='
    for i, height in enumerate([0.1, 0.8]):#0.5, 1.0, 1.5]):
        slc = yt.SlicePlot(ds, 'z', plotvar, axes_unit='Mm',
                      center=[1.0, 1.0, height]*u.Mm)
        slc.zoom(2)
        slc.set_log(plotvar, False)
        slc.set_cmap(plotvar, 'viridis')#'coolwarm')
        #slc.set_zlim(plotvar, -1.5e-5, 1.5e-5)
        slc.set_zlim(plotvar, -extreme, extreme)
        slc.annotate_velocity(scale=extremev.value*10)#normalize=True)
        slc.annotate_title('Height = {} Mm'.format(str(height)))
        slc.annotate_timestamp(text_args={'color': 'black'})

        plot = slc.plots[plotvar]
        plot.figure = fig
        plot.axes = grid[i].axes
        if i == 1: plot.cax = grid.cbar_axes[0]
        slc._setup_plots()

    """tr = Affine2D().scale(1, 1)
    grid_helper = floating_axes.GridHelperCurveLinear(
        tr, extremes=(0, 1, 0, len(ts)))
    tbar = floating_axes.FloatingSubplot(fig, 111, grid_helper=grid_helper)
    tbar.bar([0], [t])"""
    plt.savefig('/fastdata/sm1ajl/Flux-Surfaces/figs/{}/density-and-velocity-xy/{:03}'.format(mode, t))
    plt.close()

    continue ##################

    """slc = yt.SlicePlot(ds, 'z', 'theta', axes_unit='Mm', center=[1.0, 1.0, 0.1]*u.Mm)
    slc.set_cmap('theta', 'coolwarm')
    slc.annotate_timestamp(text_args={'color': 'black'})
    slc.save('figs/{}/theta/{:03}'.format(mode, t))
  
    slc = yt.SlicePlot(ds, 'z', 'v_theta', axes_unit='Mm', center=[1.0, 1.0, 0.1]*u.Mm)
    slc.set_cmap('v_theta', 'coolwarm')
    slc.annotate_timestamp(text_args={'color': 'black'})
    slc.save('figs/{}/v-theta/{:03}'.format(mode, t))

    slc = yt.SlicePlot(ds, 'z', 'v_r', axes_unit='Mm', center=[1.0, 1.0, 0.1]*u.Mm)
    slc.set_cmap('v_r', 'coolwarm')
    slc.annotate_timestamp(text_args={'color': 'black'})
    slc.save('figs/{}/v-r/{:03}'.format(mode, t))"""

"""  for t in range(0, len(ts), 2):
    ds = ts[t]
    slc = yt.SlicePlot(ds, 'x', plotvar, origin='lower-center-domain',
                      axes_unit='Mm')
    slc.set_log(plotvar, False)
    slc.set_cmap(plotvar, 'coolwarm')
    #slc.set_zlim(plotvar, -5e-6, 5e-6)
    slc.set_zlim(plotvar, -extreme, extreme)

    seed_points = np.zeros([11,2]) + 1.52
    seed_points[:,0] = np.linspace(-0.99, 0.95, seed_points.shape[0],
                                   endpoint=True)

    minb, maxb = ds.all_data().quantities.extrema("magnetic_field_strength")
    norm = mpl.colors.LogNorm(minb.value+1e-5, maxb.value)
    slc.annotate_streamlines('mag_field_y', 'mag_field_z',
                             field_color='magnetic_field_strength',
                             plot_args={'start_points': seed_points,
                                        'density': 15,
                                        'cmap': 'plasma', 'linewidth':2,
                                        'norm':norm
                                        })
    
    slc.annotate_velocity(scale=extremev.value*10)#scale=2000000)#normalize=True)
    slc.annotate_timestamp(text_args={'color': 'black'})
    slc.save('figs/{}/density-and-velocity-vs-height/{:03}'.format(mode, t))"""

print "\n\n=====\nPlots complete\n=====\n\n"
