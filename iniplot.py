import yt
import pysac.yt
import matplotlib.pyplot as plt
import astropy.units as u

thisfile = yt.load('/fastdata/sm1ajl/driver-widths/gdf/Slog_p90-0_4-8_0-30/Slog_p90-0_4-8_0-30_00001.gdf')
#print thisfile.field_list

"""for field in thisfile.field_list:
    myplot = yt.SlicePlot(thisfile, 'x', field, axes_unit='m')
    myplot.set_cmap(field, 'viridis')
    myplot.save('/fastdata/sm1ajl/initestplots/')"""

beta = 'magnetic_field_strength'

betaplot = yt.SlicePlot(thisfile, 'x', beta, axes_unit='Mm')#, center=[0.0, 0.0, 0.0]*u.Mm)
betaplot.set_log(beta, False)
#betaplot.set_zlim(beta, 0.1, 10)
#betaplot.set_zlim(beta, 0.9, 1.1)
#betaplot.zoom(2)
betaplot.annotate_contour('plasma_beta', take_log=False, ncont=1, label=True, clim=(1, 1))
betaplot.set_cmap(beta, 'coolwarm')
betaplot.save('/fastdata/sm1ajl/initestplots/')

print(thisfile.all_data().quantities.extrema(beta),
      betaplot.data_source[beta].mean())
print thisfile.domain_dimensions


betaplot = yt.SlicePlot(thisfile, 'z', beta, axes_unit='Mm')#, center=[0.0, 0.0, -0.8]*u.Mm)
betaplot.set_log(beta, False)
#betaplot.set_zlim(beta, 0.1, 10)
#betaplot.set_zlim(beta, 0.9, 1.1)
#betaplot.zoom(2)
#betaplot.annotate_contour('plasma_beta', take_log=False, ncont=1, label=True, clim=(1, 1))
betaplot.set_cmap(beta, 'coolwarm')
betaplot.save('/fastdata/sm1ajl/initestplots/')

print(thisfile.all_data().quantities.extrema(beta),
      betaplot.data_source[beta].mean())
print thisfile.domain_dimensions
