import yt
import yt.units as u
import pysac.yt
import matplotlib.pyplot as plt
import numpy as np
import sys

r0 = 0.1e6 * u.Mm
period = 120
global t
t = 30


def tdep():
    if t < period / 2.0:
        return np.sin(t* 2.0 * np.pi / period)
    else:
        return 0.0


def _alfvenspeed(field, data):
    bx, by, bz = data['mag_field_x_bg'], data['mag_field_y_bg'], data['mag_field_z_bg']
    B = np.sqrt(bx**2 + by**2 + bz**2)
    return u.yt_array.YTArray(B / np.sqrt(data['density_bg']), 'Mm/s')


def _theta(field, data):
    x, y = data['x'], data['y']
    return u.yt_array.YTArray(np.arctan2(y, x), 'radian')


def _r(field, data):
    x, y = data['x'], data['y']
    return u.yt_array.YTArray(np.sqrt(x**2 + y**2), 'Mm')


def _v_theta(field, data):
    v_A, r, theta = data['alfvenspeed'], data['r'] / r0, data['theta']
    return u.yt_array.YTArray(r * (1 - r**2) * np.exp(-(r**2)) * np.cos(l * theta) * tdep(), 'Mm/s')


def _v_r(field, data):
    v_A, r, theta = data['alfvenspeed'], data['r'] / r0, data['theta']
    return u.yt_array.YTArray(l * (r / 2.0) * np.exp(-(r**2)) * np.sin(l * theta) * tdep(), 'Mm/s')


def _v_x(field, data):
    u_r, u_th, theta = data['v_r'], data['v_theta'], data['theta']
    return u.yt_array.YTArray((u_r * np.cos(theta)) - (u_th * np.sin(theta)), 'Mm/s')


def _v_y(field, data):
    u_r, u_th, theta = data['v_r'], data['v_theta'], data['theta']
    return u.yt_array.YTArray((u_r * np.sin(theta)) + (u_th * np.cos(theta)), 'Mm/s')


yt.add_field('alfvenspeed', function=_alfvenspeed, units='Mm/s')
yt.add_field('theta', function=_theta, units='radian')
yt.add_field('r', function=_r, units='Mm')
yt.add_field('v_theta', function=_v_theta, units='Mm/s')
yt.add_field('v_r', function=_v_r, units='Mm/s')
yt.add_field('v_x', function=_v_x, units='Mm/s')
yt.add_field('v_y', function=_v_y, units='Mm/s')

l = 0
#thisfile = yt.load('/data/sm1ajl/mhs_atmosphere/drew_model/drew_model.gdf')
thisfile = yt.load('/fastdata/sm1ajl/Flux-Surfaces/gdf/m0_p120-0_0-5_0-5/m0_p120-0_0-5_0-5_00001.gdf')
#thisfile = yt.load('/data/sm1ajl/mhs_atmosphere/mfe_setup/mfe_setup.gdf')
#thisfile = yt.load('/data/sm1ajl/mhs_atmosphere/drew_paper1/drew_paper1.gdf')
ad = thisfile.all_data()
print 'B!', ad.quantities.extrema(["magnetic_field_strength"]).to_equivalent('gauss', 'CGS')
print ad.quantities.extrema(["density_bg"])
print ad.quantities.extrema(["alfvenspeed"])
print ad.quantities.extrema(["plasma_beta"])
#print ad.quantities.extrema(["v_x"])
#print ad.quantities.extrema(["v_y"])
#print ad.quantities.extrema(["v_r"])
#print ad.quantities.extrema(["v_theta"])

myplot = yt.SlicePlot(thisfile, 'x', 'plasma_beta', axes_unit='Mm', origin='lower-center-domain')
myplot.set_cmap('plasma_beta', 'viridis')
myplot.annotate_streamlines('mag_field_y', 'mag_field_z')
myplot.save('/fastdata/sm1ajl/Flux-Surfaces/figs/driverplots/m{}/'.format(l))
plt.close()

myplot = yt.SlicePlot(thisfile, 'x', 'magnetic_field_strength', axes_unit='Mm',
                      origin='lower-center-domain')
myplot.set_cmap('magnetic_field_strength', 'plasma')
Bstr = myplot.data_source['magnetic_field_strength']
print('B!', Bstr.min(), Bstr.max())
myplot.save('/fastdata/sm1ajl/Flux-Surfaces/figs/driverplots/m{}/'.format(l))
plt.close()

myplot = yt.SlicePlot(thisfile, 'z', 'plasma_beta', axes_unit='Mm',
                     center=[0.0, 0.0, 0.1])
#myplot.set_zlim('plasma_beta', 0, 2)
#myplot.set_log('plasma_beta', False)
myplot.set_cmap('plasma_beta', 'PiYG')
myplot.annotate_quiver('v_x', 'v_y', scale=7.5)
myplot.annotate_contour('magnetic_field_strength', take_log=False, ncont=1, label=True,
                        clim=(Bstr.max()/2, Bstr.max()/2))
#myplot.zoom(2)
myplot.save('/fastdata/sm1ajl/Flux-Surfaces/figs/driverplots/m{}/'.format(l))
plt.close()

#print thisfile.index.grids[0]['density_bg'].min(), thisfile.index.grids[0]['density_bg'].max()
#total = thisfile.index.grids[0]['density_bg'] + thisfile.index.grids[0]['density_pert']
#print total.min(), total.max()

sys.exit()
for m in range(4):
    print 'm = ', m
    v_A = 1.0
    alpha = 1.0
    tdep = 1.0

    xx = np.linspace(-1.0e6, 1.0e6, 256)
    yy = np.linspace(-1.0e6, 1.0e6, 256)
    zz = np.linspace(0, 1.6e6, 128)

    x, y = np.meshgrid(xx, yy)

    r0 = 0.5e6
    r = np.sqrt(x**2 + y**2) / r0
    theta = np.arctan2(y, x)

    exp_z = 1.0
    fig, ax = plt.subplots(1, 2, figsize=(24, 12))
    fig.patch.set_facecolor('white')

    for i, l in enumerate([-m, m]):
        print 'm = ', l
        u_th = v_A * alpha * r * (1 - r**2) * np.exp(-(r**2)) * np.cos(l * theta)
        u_r = v_A * alpha * l * (r / 2.0) * np.exp(-(r**2)) * np.sin(l * theta)

        ux = ((u_r * np.cos(theta)) - (u_th * np.sin(theta)))[::4, ::4]
        uy = ((u_r * np.sin(theta)) + (u_th * np.cos(theta)))[::4, ::4]

        u = np.sqrt(ux**2 + uy**2)

        ax[i].imshow(u, cmap='gist_gray', origin='lower', extent=[xx[0], xx[-1], yy[0], yy[-1]])
        ax[i].quiver(x[::4, ::4], y[::4, ::4], ux, uy, scale=8.0, color='gray')
        ax[i].set_title('m = {}'.format(l))
    plt.savefig('/fastdata/sm1ajl/Flux-Surfaces/drivershape_m={}'.format(m),
               facecolor='white', transparent=False)
    plt.close()
