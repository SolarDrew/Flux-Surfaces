#from matplotlib import use
#use('pdf')
import yt
import pysac.yt
import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np

#thisfile = yt.load('/fastdata/sm1ajl/driver-widths/gdf/Slog_p90-0_4-8_0-30/Slog_p90-0_4-8_0-30_00001.gdf')
#thisfile = yt.load('/data/sm1ajl/custom_ini_files/paper2b/paper2b.gdf')
#thisfile = yt.load('/data/sm1ajl/custom_ini_files/drew_paper1/drew_paper1.gdf')
thisfile = yt.load('/data/sm1ajl/custom_ini_files/drew_model/drew_model.gdf')
#print thisfile.field_list

"""for field in thisfile.field_list:
    myplot = yt.SlicePlot(thisfile, 'x', field, axes_unit='m')
    myplot.set_cmap(field, 'viridis')
    myplot.save('/fastdata/sm1ajl/initestplots/')"""

beta = 'density_bg'
#beta = 'plasma_beta'
#beta = 'magnetic_field_strength'
#beta = 'thermal_pressure'

betaplot = yt.SlicePlot(thisfile, 'x', beta, axes_unit='Mm', origin='native')#, center=[0.0, 0.0, 0.0]*u.Mm)
#minvar, maxvar = betaplot.data_source[beta].min(), betaplot.data_source[beta].max()
betaplot.set_log(beta, False)
#betaplot.set_zlim(beta, 10**-2, 10**4.7)
#betaplot.set_cmap(beta, 'YlOrBr')
#betaplot.set_cmap(beta, 'coolwarm')
#betaplot.set_cmap(beta, 'PiYG')
betaplot.set_cmap(beta, 'viridis')
#betaplot.set_colorbar_label(beta, 
#    'Background density $\\left(\\frac{\\mathrm{kg}}{\\mathrm{m}^3}\\right)$')
#betaplot.set_zlim(beta, 10**-2.3, 10**5.78)
betaplot.annotate_contour('plasma_beta', take_log=False, label=True,# clim=(10**-1.44, 10**5.78),
                          plot_args={'cmap': 'PiYG', 'levels': [0.02, 0.05, 0.1, 1.0, 10, 100, 1000]})
seed_points = np.zeros([11, 2]) + 1.45 #+ 2.5
seed_points[:, 0] = np.linspace(-0.99, 0.95, seed_points.shape[0],
#seed_points[:, 0] = np.linspace(-0.45, 0.45, seed_points.shape[0],
                                endpoint=True)
betaplot.annotate_streamlines('mag_field_y_bg', 'mag_field_z_bg',
                              field_color='magnetic_field_strength_bg',
                              plot_args={'start_points': seed_points,
                                         'density': 15,
                                         #'cmap': 'inferno_r', 'linewidth':2,
                                         'color': 'black', 'linewidth': 1,
                                         })
betaplot.save('/data/sm1ajl/Flux-Surfaces/ini_conditions_2D')

print(thisfile.all_data().quantities.extrema(beta),
      betaplot.data_source[beta].mean())
print thisfile.domain_dimensions

for height in [0.5]:#range(9):
    betaplot = yt.SlicePlot(thisfile, 'z', beta, axes_unit='Mm', center=[0.0, 0.0, height]*u.Mm, origin='native')
    betaplot.set_log(beta, False)
    betaplot.annotate_streamlines('mag_field_x', 'mag_field_y', plot_args={'density': 1.5})
    #betaplot.set_zlim(beta, 154, 157)
    #betaplot.annotate_contour('plasma_beta', take_log=False, ncont=1, label=True, clim=(1, 1))
    betaplot.set_cmap(beta, 'YlOrBr')
    betaplot.save('/data/sm1ajl/Flux-Surfaces/ini_conditions_2D_{}Mm'.format(height).replace('.', '_'))
    
    print(thisfile.all_data().quantities.extrema(beta),
          betaplot.data_source[beta].mean())
    print thisfile.domain_dimensions
