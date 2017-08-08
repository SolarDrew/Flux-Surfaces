from __future__ import print_function
import os
import sys
import glob
import numpy as np
from matplotlib import use, rcParams
use('pdf')
rcParams['font.size'] = 32
import matplotlib.pyplot as plt
from sacconfig import SACConfig
import yt
from yt.funcs import mylog
mylog.setLevel(50)

m = 0
os.system('./configure.py set SAC --usr_script=m{}'.format(m))
cfg = SACConfig()

# Height slice(s) with distance-displacement plots
radii = range(3, 64, 3)
heights = range(15, 125, 50)

alldists = np.zeros((len(radii)*2, len(heights), 187, 100))
radcoords = np.zeros((len(radii)*2, len(heights)))

fname = glob.glob(os.path.join(cfg.data_dir, 'Times*.npy'))
times = np.load(fname[0])

for j, r in enumerate(radii):
    tube_r = 'r{:02}'.format(r)
    for i, h in enumerate(heights):
        try:
            dis = np.load(
                os.path.join(cfg.data_dir, tube_r,
                             "{}_h{}_distance.npy".format(cfg.get_identifier(),
                                                          h))
            )[:187]
            n_t, n_theta = dis.shape
            radcoords[j*2, i] = dis.mean()
            d1 = dis - dis[0][None]
            d1 *= 15.6e3

            alldists[j*2, i, :, :] = d1
        except IOError:
            print('Failed radius {}, height {}'.format(r, h), file=sys.stderr)
        except IndexError:
            print('Failed radius {}, height {}'.format(r, h), file=sys.stderr)

sep = 1.04
radcoords[1::2] = radcoords[::2] * sep
radlims = [600, 800, 1000]

#for time in range(1, 187, 1):
for time in [150]:
    ds = yt.load(
        '/fastdata/sm1ajl/Flux-Surfaces/gdf/{0}/{0}_{1:05}.gdf'.format(
            cfg.get_identifier(), time))
    simtime = ds.current_time.to('s')


    for i, h in enumerate(heights):
        height = h * 12.5
        hslice = alldists[:, i, time, :]
        n_r, n_theta = hslice.shape
        sym_max = np.max((np.abs(np.min(hslice)), np.max(hslice)))
        vmin = sym_max * -1.
        vmax = sym_max

        fig = plt.figure(figsize=(18, 12)) 
        ax = fig.add_subplot(1, 1, 1, projection='polar')
        R, T = np.meshgrid(radcoords[:, i]*15.6,
                           np.linspace(0, 2*np.pi, n_theta))
        c = ax.pcolormesh(T, R, hslice.T, rasterized=True, cmap='RdBu_r',
                          vmin=vmin, vmax=vmax)
        maxrad = radlims[i]
        ticks = [(tik//10)*10 for tik in np.linspace(0, maxrad, 5)]
        ax.set_rmax(maxrad)
        cbar = plt.colorbar(c, ax=ax, shrink=0.9, pad=0.1)
        cbar.ax.set_ylabel('Radial Displacement [m]')
        fig.text(0.05, 0.9, '$z = {}$ km'.format(height))#, fontsize=28)

        ax.set_yticks(ticks)
        ax.set_yticklabels([])
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.grid()

        # tick locations
        thetaticks = np.arange(0, 360, 45)
        # set ticklabels location at 1.3 times the axes' radius
        ax.set_thetagrids(thetaticks, frac=1.2)#, fontsize=16)

        ax.set_xlabel("Angle around flux surface [degrees]")#, fontsize=20)

        plt.savefig('figs/poster/slices/dr-hslice_m{}_h{}'.format(m, h).replace('.', '-'), bbox_inches='tight')
        plt.close()
