from __future__ import division
import numpy as np
from itertools import product
import sys
from astropy import units as u
from sacconfig import SACConfig
import fredatmos2 as atm


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

outname = '/fastdata/sm1ajl/inidata/fredatmos_{}.ini'.format(mpic)
header0 = 'fred-atmosphere_mhd33\n'
n_dims = 3
dims = ['x', 'y', 'z']
vars = ['h', 'm1', 'm2', 'm3', 'e', 'b1', 'b2', 'b3', 'eb', 'rhob', 'bg1', 'bg2', 'bg3']
varnames = {'rhob': 'density',
            'eb': 'energy',
            'bg1': 'magnetic_field_z',
            'bg2': 'magnetic_field_x',
            'bg3': 'magnetic_field_y'}
eqpars = ['gamma', 'eta', 'grav0', 'grav1', 'grav2', 'grav3', 'nu']
#eqparvals = [1.4, 0.0, 1.0, -274.0, 0.0, 1.0, 0.0]
pars0 = ini0.parameters
eqparvals = [pars0['gamma'], pars0['eta'], pars0['gravity0'], pars0['gravity1'], pars0['gravity2'], 1.0, pars0['nu']]

for nam, dat in atm.data.iteritems():
    print nam, dat.shape, dat.min()

# Define domain
full_domain_size = [int(i) for i in cfg.grid_size.split(',')] #[120, 120, 120]
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
print atm.x.shape

bz, bx, by = (atm.data[varnames['bg1']].to_equivalent('gauss', 'CGS'),
              atm.data[varnames['bg2']].to_equivalent('gauss', 'CGS'),
              atm.data[varnames['bg3']].to_equivalent('gauss', 'CGS'))
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
    print '=', procid, '=', atm.x[arr_slice].shape
    """x = np.rollaxis(atm.x, 2, 0)[arr_slice].flatten().to(u.m)
    y = np.rollaxis(atm.y, 2, 0)[arr_slice].flatten().to(u.m)
    z = np.rollaxis(atm.z, 2, 0)[arr_slice].flatten().to(u.m)"""
    x = atm.x[arr_slice].flatten().to(u.m)
    y = atm.y[arr_slice].flatten().to(u.m)
    z = atm.z[arr_slice].flatten().to(u.m)

    # Arrange values in array for output
    outdata = np.zeros(shape=(prod(domain_size), len(dims + vars)))
    #print '--', outdata.shape
    for i, thisdim in enumerate([z, x, y]):
        outdata[:, i] = thisdim
    
    for thisvar in varnames:
        thisdata = atm.data[varnames[thisvar]]
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
