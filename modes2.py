# coding: utf-8
#import pysac.analysis.tube3D.process_utils as utils
import yt
import pysac.yt
import yt.units as u
from yt.visualization.api import get_multi_plot
import numpy as np
import matplotlib as mpl
import matplotlib.animation as ani
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import sys
from yt.utilities.physical_constants import mu_0
from os import makedirs
from os.path import exists, join


def _theta(field, data):
    x, y = data['x'], data['y']
    return u.yt_array.YTArray(np.arctan2(y, x), 'radian')


def _velocity_rot(field, data):
    x, y, theta = data['velocity_x'], data['velocity_y'], data['theta']
    return u.yt_array.YTArray(-x*np.sin(theta) + y*np.cos(theta), x.units)


def _velocity_rot_abs(field, data):
    return u.yt_array.YTArray(abs(data['velocity_rot']), data['velocity_rot'].units)


def _velocity_r(field, data):
    x, y, theta = data['velocity_x'], data['velocity_y'], data['theta']
    return u.yt_array.YTArray(x*np.cos(theta) - y*np.sin(theta), x.units)


def _alfvenspeed(field, data):
    bx = data['mag_field_x_bg'] + data['mag_field_x_pert']
    by = data['mag_field_y_bg'] + data['mag_field_y_pert']
    bz = data['mag_field_z_bg'] + data['mag_field_z_pert']
    B = np.sqrt(bx**2 + by**2 + bz**2)

    return u.yt_array.YTArray(B / np.sqrt(data['density_bg'] + data['density_pert']), 'm/s')


def _thermal_pressure_pert(field, data):
    #p = (\gamma -1) ( e - \rho v^2/2 - B^2/2)
    g1 = data.ds.parameters.get('gamma', 5./3.) -1
    if data.ds.dimensionality == 2:
        kp = (data['density'] * (data['velocity_x']**2 +
                                 data['velocity_y']**2))/2.
    if data.ds.dimensionality == 3:
        kp = (data['density'] * (data['velocity_x']**2 +
              data['velocity_y']**2 + data['velocity_z']**2))/2.
    return g1 * (data['internal_energy_pert'] - kp - data['mag_pressure_pert'])


def _thermal_pressure_bg(field, data):
    return data['thermal_pressure'] - data['thermal_pressure_pert']


def _thermal_pressure_frac(field, data):
    return data['thermal_pressure_pert'] / data['thermal_pressure_bg']


def _density_frac(field, data):
    return data['density_pert'] / data['density_bg']


def _mag_pressure_pert(field, data):
    if data.ds.dimensionality == 2:
        Bp_sq_2 = (data['mag_field_x_pert']**2 +
                   data['mag_field_y_pert']**2) / (2. * mu_0)
    if data.ds.dimensionality == 3:
        Bp_sq_2 = (data['mag_field_x_pert']**2 + data['mag_field_y_pert']**2 +
                   data['mag_field_z_pert']**2) / (2. * mu_0)
    return (((data['magnetic_field_strength_bg'] * data['magnetic_field_strength_pert']) / mu_0) 
            + Bp_sq_2)# / np.sqrt(mu_0)


def _scaled_speed(field, data):
    return data['velocity_magnitude'] / data['alfvenspeed']


yt.add_field('theta', function=_theta, units='radian')
yt.add_field('velocity_rot', function=_velocity_rot, units='m/s')
yt.add_field('velocity_rot_abs', function=_velocity_rot_abs, units='m/s')
yt.add_field('velocity_r', function=_velocity_r, units='m/s')
yt.add_field('alfvenspeed', function=_alfvenspeed, units='m/s')
yt.add_field('thermal_pressure_pert', function=_thermal_pressure_pert, units='Pa')
yt.add_field('thermal_pressure_bg', function=_thermal_pressure_bg, units='Pa')
yt.add_field('thermal_pressure_frac', function=_thermal_pressure_frac, units='dimensionless')
yt.add_field('density_frac', function=_density_frac, units='dimensionless')
yt.add_field('mag_pressure_pert', function=_mag_pressure_pert, units='Pa')
yt.add_field('scaled_speed', function=_scaled_speed, units='dimensionless')

plotvar, t = sys.argv[1:]
#Blim = [0.005, 0.002, 0.0009, 0.00062]

savedir = join('/fastdata', 'sm1ajl', 'Flux-Surfaces', 'figs')
for mode in ['m0', 'm1', 'm-1', 'm2', 'm-2']:
  #ts = yt.load('/fastdata/sm1ajl/Flux-Surfaces/gdf/{0}_p60-0_0-5_0-5/{0}_p60-0_0-5_0-5_00???.gdf'.format(mode))
    print(mode, t)
    print('/fastdata/sm1ajl/Flux-Surfaces/gdf/{0}_p60-0_0-5_0-5/{0}_p60-0_0-5_0-5_00{1:03}.gdf')
    ds = yt.load('/fastdata/sm1ajl/Flux-Surfaces/gdf/{0}_p60-0_0-5_0-5/{0}_p60-0_0-5_0-5_00{1:03}.gdf'.format(mode, int(t)))
  #print len(ts)

  #for t in range(0, 27):#10):#2):
    #ds = ts[t]

    """
    Plot vertical slice at various heights
    """
    fig = plt.figure()#figsize=(15, 8))
    #grid = AxesGrid(fig, (0.075,0.075,0.85,0.85), nrows_ncols=(1, 3),
    #                cbar_location="right", cbar_mode="each", cbar_size="4%", cbar_pad="2%",
    #                axes_pad=1.0)
    grid = AxesGrid(fig, (0.075,0.075,0.85,0.85), nrows_ncols=(2, 2),
                    cbar_location="right", cbar_mode="each", cbar_size="4%", cbar_pad="2%",
                    axes_pad=1.0)
    #grid = AxesGrid(fig, (0.075,0.075,0.85,0.85), nrows_ncols=(1, 2),
    #                cbar_location="right", cbar_mode="edge", cbar_size="5%", cbar_pad="0%")

    minv, maxv = ds.all_data().quantities.extrema("velocity_magnitude").to('m/s')

    minvar, maxvar = ds.all_data().quantities.extrema(plotvar)
    print t, "====", minv, maxv, '===='
    if 'velocity' in plotvar:
        minvar = minvar.to('m/s')
        maxvar = maxvar.to('m/s')
    #if plotvar == 'velocity_z':
    #    extreme = 0.5 * u.km / u.s
    #else:
    extreme = max(abs(minvar), abs(maxvar))
    extremev = max(abs(minv), abs(maxv))

    if 'mag' in plotvar and 'pressure' not in plotvar and 'velocity' not in plotvar:
        print '====', minvar.to_equivalent('gauss', 'CGS'), maxvar.to_equivalent('gauss', 'CGS'), '===='
    else:
        print '====', minvar, maxvar, '===='

    #for i, height in enumerate([0.1, 0.8]):
    #for i, height in enumerate([0.1, 0.5, 1.0, 1.5]):
    for i, height in enumerate([0.05, 0.15, 0.7, 1.4]):
        slc = yt.SlicePlot(ds, 'z', plotvar, axes_unit='Mm',
                      center=[0.0, 0.0, height]*u.Mm)
        #slc.zoom(2)
        minvar, maxvar = slc.data_source[plotvar].min(), slc.data_source[plotvar].max()
        if 'vel' in plotvar:
            print 'minv maxv ::::', minvar.to('m/s'), maxvar.to('m/s')
        #print height, minvar, maxvar, slc.data_source[plotvar].std()
        #extreme = slc.data_source[plotvar].std() * 3
        #extreme = (abs(minvar) + abs(maxvar)) / 2. #4.
        #extreme = absvar.mean() + (3*absvar.std())
        #linthresh = absvar.mean() * 0.5# - (0.5*absvar.std())
        #print('+=+=+', linthresh, '+=+=+')
        var = slc.data_source[plotvar]
        if 'velocity' in plotvar or 'v_' in plotvar:
            slc.set_unit(plotvar, 'm/s')
            extreme = extreme.to('m/s')
            extreme = max(abs(minvar).to('m/s'), abs(maxvar).to('m/s'))
            absvar = abs(slc.data_source[plotvar])
        else:
            extreme = max(abs(minvar), abs(maxvar))
            absvar = abs(slc.data_source[plotvar])
        print extreme
        if '_x' in plotvar \
          or '_y' in plotvar \
          or '_z' in plotvar \
          or '_pert' in plotvar:
            slc.set_log(plotvar, True, linthresh=1e-11)
            slc.set_cmap(plotvar, 'coolwarm')
            #slc.set_cmap(plotvar, 'seismic')
            slc.set_zlim(plotvar, -extreme, extreme)
        elif '_frac' in plotvar:
            slc.set_log(plotvar, True, linthresh=1e-5)#linthresh)
            slc.set_cmap(plotvar, 'BrBG')
            slc.set_zlim(plotvar, -extreme, extreme)
        elif '_rot' in plotvar:
            slc.set_log(plotvar, False)
            slc.set_cmap(plotvar, 'RdYlGn')
            slc.set_zlim(plotvar, -extreme, extreme)
            slc.set_colorbar_label(plotvar, 'Rotational velocity (m/s)')
        elif plotvar == 'magnetic_field_strength':
            slc.set_cmap(plotvar, 'coolwarm')
            slc.set_zlim(plotvar, 0, maxvar.value)
        elif plotvar == 'plasma_beta':# \
        #  or '_pressure' in plotvar:
            slc.set_cmap(plotvar, 'magma')
            slc.annotate_contour(plotvar, take_log=False, ncont=1, label=True, clim=(1, 1))
        elif plotvar == 'temperature':
            slc.set_cmap(plotvar, 'inferno')
        else:
            slc.set_cmap(plotvar, 'viridis')
            slc.set_zlim(plotvar, 0, maxvar.value)
            slc.set_log(plotvar, False)
        if plotvar == 'density_bg':
            slc.annotate_contour('density_pert', take_log=False, label=True)
        else:
            slc.annotate_contour('plasma_beta', take_log=False, label=True, plot_args={'levels': [1, 10, 100, 1000, 10000]}, text_args={'fmt': '%i'})
        slc.annotate_velocity(scale=0.8e5)
        #if plotvar != 'magnetic_field_strength':
        #for i in dir(slc.data_source['magnetic_field_strength']): print i
        """minB = slc.data_source['magnetic_field_strength'].min()
        maxB = slc.data_source['magnetic_field_strength'].max()"""
        #meanB = slc.data_source['magnetic_field_strength'].max()/2#mean()
        ##print minB, meanB, maxB
        #slc.annotate_contour('magnetic_field_strength', plot_args={'color': 'black'},#'cmap': 'inferno_r'},
        #        take_log=False, ncont=1, label=True, clim=(meanB, meanB))
        slc.annotate_title('Height = {} Mm'.format(str(height)))
        slc.annotate_timestamp(text_args={'color': 'black'})
        """if plotvar == 'density':# \
        #  or plotvar == 'temperature':
            slc.set_log(plotvar, True)
        else:"""
        #slc.set_log(plotvar, False)

        plot = slc.plots[plotvar]
        plot.figure = fig
        plot.axes = grid[i].axes
        #if i == 3: plot.cax = grid.cbar_axes[0]
        plot.cax = grid.cbar_axes[i]
        slc._setup_plots()

    thisdir = join(savedir, mode, 'density-and-velocity-xy', plotvar)
    if not exists(thisdir): makedirs(thisdir)
    slc.save(join(thisdir, '{:03}'.format(int(t))))
    plt.close()
    #continue

    slicedr = 'y'
    notslice = 'x'
    fig = plt.figure()#figsize=(18, 8))
    #grid = AxesGrid(fig, (0.075,0.075,0.85,0.85), nrows_ncols=(1, 2),
    #                cbar_location="right", cbar_mode="each", cbar_size="5%", cbar_pad="0%",
    #                axes_pad=1.0)
    
    for i, depth in enumerate([0.0]):#[-0.95, 0.0]):
        if slicedr == 'y':
            ds.coordinates.x_axis[1] = 0
            ds.coordinates.x_axis['y'] = 0
            ds.coordinates.y_axis[1] = 2
            ds.coordinates.y_axis['y'] = 2
            slc = yt.SlicePlot(ds, slicedr, plotvar, axes_unit='Mm', origin='native',
                               center=[0.0, depth, 0.8]*u.Mm)
            slc.set_xlabel('x (Mm)')
            slc.set_ylabel('z (Mm)')
        else:
            slc = yt.SlicePlot(ds, slicedr, plotvar, axes_unit='Mm', origin='native',
                               center=[depth, 0.0, 0.8]*u.Mm)
        minvar, maxvar = slc.data_source[plotvar].min(), slc.data_source[plotvar].max()
        if 'vel' in plotvar:
            print depth, minvar.to('m/s'), maxvar.to('m/s'), slc.data_source[plotvar].std().to('m/s')
        #extreme = slc.data_source[plotvar].std() * 3
        #extreme = (abs(minvar) + abs(maxvar)) / 4. #2.
        #extreme = absvar.mean() + (3*absvar.std())
        #linthresh = absvar.mean() * 0.01# - (0.5*absvar.std())
        #print('+=+=+', linthresh, '+=+=+')
        """if plotvar == 'plasma_beta':# \
        #  or plotvar == 'thermal_pressure_pert':
            slc.set_log(plotvar, True)
        else:
            slc.set_log(plotvar, False)"""
        if 'velocity' in plotvar:
            slc.set_unit(plotvar, 'm/s')
            extreme = extreme.to('m/s')
            extreme = max(abs(minvar).to('m/s'), abs(maxvar).to('m/s'))
            absvar = abs(slc.data_source[plotvar].to('m/s'))
        else:
            extreme = max(abs(minvar), abs(maxvar))
            absvar = abs(slc.data_source[plotvar])
        if '_x' in plotvar \
          or '_y' in plotvar \
          or '_z' in plotvar \
          or '_pert' in plotvar:
            slc.set_log(plotvar, True, linthresh=1e-11)
            slc.set_cmap(plotvar, 'coolwarm')
            #slc.set_cmap(plotvar, 'seismic')
            #slc.set_cmap(plotvar, 'RdBu_r')
            slc.set_zlim(plotvar, -extreme, extreme)
        elif '_frac' in plotvar:
            slc.set_log(plotvar, True, linthresh=1e-4)#linthresh)
            slc.set_cmap(plotvar, 'BrBG')
            slc.set_zlim(plotvar, -extreme, extreme)
        elif '_rot' in plotvar:
            slc.set_log(plotvar, False)
            slc.set_cmap(plotvar, 'RdYlGn')
            slc.set_zlim(plotvar, -extreme, extreme)
            slc.set_colorbar_label(plotvar, 'Rotational velocity (m/s)')
        elif plotvar == 'magnetic_field_strength':
            slc.set_cmap(plotvar, 'coolwarm')
            slc.set_zlim(plotvar, 0, maxvar.value)
        elif plotvar == 'plasma_beta':# \
        #  or '_pressure' in plotvar:
            slc.set_cmap(plotvar, 'magma')
            slc.annotate_contour(plotvar, take_log=False, ncont=1, label=True, clim=(1, 1))
        elif plotvar == 'temperature':
            slc.set_cmap(plotvar, 'inferno')
        elif 'scaled' in plotvar:
            slc.set_cmap(plotvar, 'cubehelix_r')
            slc.set_log(plotvar, False)
        else:
            slc.set_cmap(plotvar, 'viridis')
            slc.set_zlim(plotvar, 0, maxvar.value)
            slc.set_log(plotvar, False)
        if plotvar == 'density_bg':
            slc.annotate_contour('density_pert', take_log=False, label=True)
        else:
            slc.annotate_contour('plasma_beta', take_log=False, label=True, plot_args={'levels': [1, 10, 100, 1000, 10000]}, text_args={'fmt': '%i'})
        #slc.annotate_contour('velocity_magnitude', take_log=False, label=True,
        #                     plot_args={'cmap': 'cubehelix_r'})

        seed_points = np.zeros([11,2]) + 1.52
        seed_points[:, 0] = np.linspace(-0.99, 0.95, seed_points.shape[0],
                                       endpoint=True)
        slc.annotate_streamlines('mag_field_'+notslice, 'mag_field_z',
                                 field_color='magnetic_field_strength',
                                 plot_args={'start_points': seed_points,
                                            'density': 15,
                                            'cmap': 'inferno_r', 'linewidth':0.5,
                                 #           'color': 'grey'
                                            })
        seed_points = np.zeros([2,2]) + 0.1
        seed_points[:, 0] = [-0.1, 0.1]
        slc.annotate_streamlines('mag_field_'+notslice, 'mag_field_z',
                                 plot_args={'start_points': seed_points,
                                            'density': 15,
                                            'color': 'black',
                                            'linewidth': 3
                                           })
        #if slicedr == 'y':
        #    slc.plots[plotvar].
    
        slc.annotate_title(r'${}$ = {} Mm'.format(slicedr, str(depth)))
        slc.annotate_timestamp(text_args={'color': 'black'})

        #plot = slc.plots[plotvar]
        #plot.figure = fig
        #plot.axes = grid[i].axes
        #if i == 1: plot.cax = grid.cbar_axes[0]
        #plot.cax = grid.cbar_axes[i]
        slc._setup_plots()

    thisdir = join(savedir, mode, 'density-and-velocity-vs-height', plotvar)
    if not exists(thisdir): makedirs(thisdir)
    slc.save(join(thisdir, '{:03}'.format(int(t))))
    plt.close()

print "\n\n=====\nPlots complete\n=====\n\n"
