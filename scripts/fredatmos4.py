from __future__ import division
import numpy as np
from itertools import product
import sys
from astropy import units as u
from sacconfig import SACConfig
#import calc_atmosphere as atm
import yt


def prod(sequence):
    """Quick and dirty function to return integer size of domain for given dimensions"""
    product = 1
    for x in sequence:
        product *= x
    return product

mu0 = 4.e-7 * np.pi * u.N / u.A**2

cfg = SACConfig()
mpic = cfg.mpi_config
print cfg.grid_size, mpic

model = 'drew_model'

# Name of the gdf file to convert.
gdfname = '/data/sm1ajl/custom_ini_files/{0}/{0}.gdf'.format(model)
# Load the file and get the data
ini0 = yt.load(gdfname)
data = ini0.all_data()

# Define name of output file
#outname = '/data/sm1ajl/custom_ini_files/singletube_{}.ini'.format(mpic)
#outname = '/fastdata/sm1ajl/inidata/fredatmos-stronger_{}.ini'.format(mpic)
outname = '/fastdata/sm1ajl/inidata/fredatmos_{}.ini'.format(mpic)
# Create stub of the ini header to add things to later on
if model == 'drew_model':
    header0 = 'fred-atmosphere_mhd33\n'
else:
    header0 = '{}_mhd33\n'.format(model)
# Number and name of dimensions
n_dims = 3
dims = ['x', 'y', 'z']
# Names of variables to save out
vars = ['h', 'm1', 'm2', 'm3', 'e', 'b1', 'b2', 'b3', 'eb', 'rhob', 'bg1', 'bg2', 'bg3']
# Mapping of SAC variable names to gdf fields
varnames = {'rhob': 'density_bg',
            'eb': 'internal_energy_bg',
            'bg1': 'mag_field_z_bg',
            'bg2': 'mag_field_x_bg',
            'bg3': 'mag_field_y_bg'}
# Set parameters for MHD equations
eqpars = ['gamma', 'eta', 'grav0', 'grav1', 'grav2', 'grav3', 'nu']
pars0 = ini0.parameters
eqparvals = [pars0['gamma'], pars0['eta'], 1.0, pars0['gravity0'], pars0['gravity1'], pars0['gravity2'], pars0['nu']]

# Stuff to deal with size of full domain
lowcoords = ini0.domain_left_edge.to('m')
hicoords = ini0.domain_right_edge.to('m')
domdims = ini0.domain_dimensions
dx = ini0.domain_width.to('m') / domdims
fullx, fully, fullz = u.Quantity(np.mgrid[lowcoords[0]:hicoords[0]:dx[0],
                                          lowcoords[1]:hicoords[1]:dx[1],
                                          lowcoords[2]:hicoords[2]:dx[2]], unit=u.m)

#for nam, dat in data.iteritems():
#    print nam, dat.shape, dat.min()

# Define domain
full_domain_size = [int(i)-4 for i in cfg.grid_size.split(',')] #[120, 120, 120]
#full_domain_size = [full_domain_size[1], full_domain_size[2], full_domain_size[0]]
#full_domain_size = atm.x.shape
#print(full_domain_size)

# Get dimensions of process distribution and fiddle some things
procs_index0 = outname.find('_np')+3
print outname
nprocs = [int(outname[i:i+2]) for i in range(procs_index0, procs_index0+(2*n_dims), 2)]
print nprocs
n0, n1, n2 = nprocs
domain_size = [int(full_domain_size[i]/nprocs[i]) for i in range(len(full_domain_size))]

print domain_size, full_domain_size
#print atm.x.shape

bz, bx, by = (data[varnames['bg1']].to_equivalent('gauss', 'CGS'),
              data[varnames['bg2']].to_equivalent('gauss', 'CGS'),
              data[varnames['bg3']].to_equivalent('gauss', 'CGS'))
print '....x', bx.min(), bx.max()
print '....y', by.min(), by.max()
print '....z', bz.min(), bz.max()
B_str = np.sqrt(bx**2 + by**2 + bz**2)
print '....B', B_str.min(), B_str.max()

for procid in range(prod(nprocs)):
    print '====', procid, '====',
    coords = np.zeros(3)
    """if procid < n0:
        coords[0] = procid
    elif (procid == n0):
        coords[1] = 1
    elif procid < (n0 + n1):
        coords[0] = procid - n0
        coords[1] = procid - n1
    else:
        coords[2] = procid // (n0 * n1)
        procid2 = procid - (coords[2] * n2)
        if procid2 < n0:
            coords[0] = procid2
        elif procid2 == n0:
            coords[1] = 1
        elif procid2 < (n0 + n0):
            coords[0] = procid2 - n0
            coords[1] = procid2 - n1"""
    coords[0] = procid % n0
    coords[1] = (procid // n0) % n1
    coords[2] = (procid // (n0 * n1)) % n2

    s = map(int, coords * domain_size)
    e = map(int, (coords+1) * domain_size)
    arr_slice = np.s_[s[1]:e[1], s[2]:e[2], s[0]:e[0]]
    #print s
    #print e
    #print arr_slice, '\n----'
    print arr_slice, coords

    # Define file preamble
    header = header0 + ' {: 6} {: .5E} {: 1} {: 1} {: 1}\n'.format(0, 0.0, n_dims, len(eqpars), len(vars))
    for x in domain_size: header += ' {}'.format(x)
    header += '\n'
    for x in eqparvals: header += ' {: .5E}'.format(x)
    header += '\n'
    for x in dims + vars + eqpars: header += '{} '.format(x)

    #print '----', domain_size, prod(domain_size), len(dims + vars)
    print '=', procid, '=', fullx[arr_slice].shape
    """x = np.rollaxis(atm.x, 2, 0)[arr_slice].flatten().to(u.m)
    y = np.rollaxis(atm.y, 2, 0)[arr_slice].flatten().to(u.m)
    z = np.rollaxis(atm.z, 2, 0)[arr_slice].flatten().to(u.m)"""
    x = fullx[arr_slice].flatten().to(u.m)
    y = fully[arr_slice].flatten().to(u.m)
    z = fullz[arr_slice].flatten().to(u.m)

    # Arrange values in array for output
    outdata = np.zeros(shape=(prod(domain_size), len(dims + vars)))
    #print '--', outdata.shape
    for i, thisdim in enumerate([z, x, y]):
        outdata[:, i] = thisdim
    
    #import pdb; pdb.set_trace()
    for thisvar in varnames:
        thisdata = data[varnames[thisvar]].reshape(domdims)
        #thisdata = np.rollaxis(thisdata, 2, 0)
        thisdata = thisdata[arr_slice].flatten()
        #print ',,,', thisvar, thisdata.min(), thisdata.max()
        if 'bg' in thisvar:
            thisdata = thisdata / np.sqrt(mu0)
        #    print ',,,', thisvar, thisdata.min(), thisdata.max()
        outdata[:, n_dims+vars.index(thisvar)] = thisdata

    # Sort ini array so first coordinate changes fastest
    outdata = outdata[np.lexsort((z, x, y))]
    
    # Output ini info
    ext = '_{:03}.ini'.format(procid)
    np.savetxt(outname.replace('.ini', ext), outdata, header=header, comments="")
