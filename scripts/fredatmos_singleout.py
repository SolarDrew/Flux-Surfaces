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


cfg = SACConfig()
mpic = cfg.mpi_config
print cfg.grid_size, mpic

outname = '/fastdata/sm1ajl/inidata/fredatmos.ini' #_{}.ini'.format(mpic)
header0 = 'fred-atmosphere_mhd33\n'
n_dims = 3
dims = ['x', 'y', 'z']
vars = ['h', 'm1', 'm2', 'm3', 'e', 'b1', 'b2', 'b3', 'eb', 'rhob', 'bg1', 'bg2', 'bg3']
varnames = {'rhob': 'density',
            'eb': 'energy',
            'bg1': 'magnetic_field_x',
            'bg2': 'magnetic_field_y',
            'bg3': 'magnetic_field_z'}
eqpars = ['gamma', 'eta', 'grav0', 'grav1', 'grav2', 'grav3', 'nu']
eqparvals = [1.4, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

for nam, dat in atm.data.iteritems():
    print nam, dat.shape, dat.mean()

# Define domain
domain_size = [int(i) for i in cfg.grid_size.split(',')] #[120, 120, 120]
print(domain_size)

# Define file preamble
header = header0 + ' {: 6} {: .5E} {: 1} {: 1} {: 1}\n'.format(0, 0.0, n_dims, len(eqpars), len(vars))
for x in domain_size: header += ' {}'.format(x)
header += '\n'
for x in eqparvals: header += ' {: .5E}'.format(x)
header += '\n'
for x in dims + vars + eqpars: header += '{} '.format(x)

print domain_size, prod(domain_size), len(dims + vars)
# Shorthands. These indices may well be wrong
x = atm.x.flatten().to(u.m)
y = atm.y.flatten().to(u.m)
z = atm.z.flatten().to(u.m)

# Arrange values in array for output
outdata = np.zeros(shape=(prod(domain_size), len(dims + vars)))
for i, thisdim in enumerate([z, x, y]):
    outdata[:, i] = thisdim

for thisvar in varnames:
    thisdata = atm.data[varnames[thisvar]]
    outdata[:, n_dims+vars.index(thisvar)] = thisdata.flatten()

# Sort ini array so first coordinate changes fastest
outdata = outdata[np.lexsort((z, x, y))]

# Output ini info
np.savetxt(outname, outdata, header=header, comments="")#, fmt='% .10E')
