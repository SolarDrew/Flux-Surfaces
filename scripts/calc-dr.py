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

# Import this repos config
# m = sys.argv[1]
# n = int(sys.argv[2])
# tube_r = 'r{:02}'.format(int(sys.argv[3]))

mylog.setLevel(40)
cube_slice = np.s_[:,:,:-5]


def glob_files(tube_r, search):
    files = glob.glob(os.path.join(cfg.data_dir,tube_r,search))
    files.sort()
    return files


m = 0
for t in range(0, 180, 1):
    try:
        #for a, tube_r in enumerate(['r12', 'r33', 'r63']):
        #rad = 3
        for a, rad in enumerate(range(3, 66, 3)):
            tube_r = 'r{:02}'.format(rad)
            inname = os.path.abspath('./original-data/m{0}/{1}/Fieldline_surface_m{0}_p60-0_0-5_0-5_00001.vtp'.format(m, tube_r))
            print(inname)
            orig_data = ttf.read_step(inname)
            if t == 0:
                rpos0 = np.array(orig_data.points) - [63, 63, 0]
            rpos1 = np.array(orig_data.points) - [63, 63, 0]
    
            dr = (rpos1 - rpos0)*15.6
            print(dr.min(), dr.max())
            outname = './surface_displacement_{}_{}_{:05d}.vtp'.format(tube_r, m, t)
            writer = ttf.PolyDataWriter(outname, orig_data)
            writer.add_array(surface_dr=dr)
            writer.write()
    except ValueError as e:
        print('Failed at time-step {}'.format(t))
        print(e)
