import yt
import pysac
import matplotlib.pyplot as plt
import numpy as np
import yt.units as u

def _alfvenspeed(field, data):
    bx = data['mag_field_x_bg'] + data['mag_field_x_pert']
    by = data['mag_field_y_bg'] + data['mag_field_y_pert']
    bz = data['mag_field_z_bg'] + data['mag_field_z_pert']
    B = np.sqrt(bx**2 + by**2 + bz**2)

    return u.yt_array.YTArray(B / np.sqrt(data['density_bg'] + data['density_pert']), 'Mm/s')

yt.add_field('alfvenspeed', function=_alfvenspeed, units='Mm/s')

thisfile = yt.load(
#    '/fastdata/sm1ajl/Flux-Surfaces/gdf/test_p120-0_0-5_0-5/test_p120-0_0-5_0-5_00001.gdf')
    '/fastdata/sm1ajl/Flux-Surfaces/gdf/m0_p120-0_0-5_0-5/m0_p120-0_0-5_0-5_00001.gdf')
#    '3D_tube_128_128_128.gdf')
print(thisfile.field_list)
ad = thisfile.all_data()
print ad.quantities.extrema(["density_bg"])
print ad.quantities.extrema(["alfvenspeed"])

for field in thisfile.field_list:
    myplot = yt.SlicePlot(thisfile, 'x', field, axes_unit='Mm')
    myplot.set_cmap(field, 'viridis')
    myplot.save('/fastdata/sm1ajl/Flux-Surfaces/figs/newatmout')
    myplot = yt.SlicePlot(thisfile, 'z', field, axes_unit='Mm')
    myplot.set_cmap(field, 'viridis')
    myplot.save('/fastdata/sm1ajl/Flux-Surfaces/figs/newatmout')

"""myplot = yt.SlicePlot(thisfile, 'x', 'density_bg', axes_unit='Mm')
myplot.set_cmap('density_bg', 'viridis')
myplot.save('/fastdata/sm1ajl/Flux-Surfaces/figs/newatmout')
myplot = yt.SlicePlot(thisfile, 'z', 'density_bg', axes_unit='Mm')
myplot.set_cmap('density_bg', 'viridis')
myplot.save('/fastdata/sm1ajl/Flux-Surfaces/figs/newatmout')

myplot = yt.SlicePlot(thisfile, 'x', 'alfvenspeed', axes_unit='Mm')
myplot.set_cmap('alfvenspeed', 'viridis')
myplot.save('/fastdata/sm1ajl/Flux-Surfaces/figs/newatmout')
myplot = yt.SlicePlot(thisfile, 'z', 'alfvenspeed', axes_unit='Mm')
myplot.set_cmap('alfvenspeed', 'viridis')
myplot.save('/fastdata/sm1ajl/Flux-Surfaces/figs/newatmout')
"""
