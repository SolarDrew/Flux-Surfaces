import yt
import matplotlib.pyplot as plt.

ini = yt.load('/data/sm1ajl/mhs_atmosphere/mfe_setup/mfe_setup.gdf')

slc = yt.SlicePlot(ini, 'y', 'density_bg', axes_unit='Mm', center=[0.0, 0.0, 0.05])

plt.savefig('figs/{}/testathing')
plt.close()
