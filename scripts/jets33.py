from __future__ import division
import numpy as np
from itertools import product
import sys
from astropy import units as u
from sacconfig import SACConfig


def prod(sequence):
    """Quick and dirty function to return integer size of domain for given dimensions"""
    product = 1
    for x in sequence:
        product *= x
    return product


cfg = SACConfig()

outname = '/fastdata/sm1ajl/inidata/jet-formation.ini'#_np020204.ini'
header0 = 'jet-formation_mhd33\n'
n_dims = 3
dims = ['x', 'y', 'z']
vars = ['h', 'm1', 'm2', 'm3', 'e', 'b1', 'b2', 'b3', 'ep', 'rhob', 'bg1', 'bg2', 'bg3']
eqpars = ['gamma', 'eta', 'grav0', 'grav1', 'grav2', 'grav3', 'nu']
eqparvals = [1.4, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

# Define domain
mpic = cfg.mpi_config
print(mpic)
print(cfg.grid_size)
full_domain_size = [120, 120, 120]#list(map(int, cfg.grid_size.split(','))) * np.array([int(mpic[2:4]), int(mpic[4:6]), int(mpic[6:])]) # 128, 128, 128
print(full_domain_size)
full_mincoords = -0.5e6, -0.5e6, 0.0
full_maxcoords = 0.5e6, 0.5e6, 1.0e6

# Define v-field shape
#delta = np.array([0.5e6, 0.5e6])
#amp = 2 * u.km / u.s
vr = 0 * u.km / u.s
vphi = 0 * u.km / u.s
#vx = 0 * u.km / u.s
#vy = 0 * u.km / u.s

# Set uniform values for parameters
gamma = eqparvals[eqpars.index('gamma')]
kB = 1.38e-23 * u.J / u.K
T = 6000 * u.K
mu0 = (np.pi * 4.0e-7) * (u.newton / (u.amp**2))
mp = 1.6726219e-27 * u.kg
ne = 2e23 / u.m**3
rho = ne * mp
pressure = (ne * kB * T).si
print pressure
gasE = pressure / (gamma - 1)
Bz0 = (1000 * u.gauss) / np.sqrt(mu0)
#Bz0 = (1000 * u.gauss) / np.sqrt(mu0)
#Bz1 = (1000 * u.gauss) / np.sqrt(mu0)
Bz1 = (500 * u.gauss) / np.sqrt(mu0)
print '---\nP = ', pressure.to(u.pascal)
print 'beta_max = ', (pressure / (Bz1**2.0)).si, '\n---'
#magE = ((gamma - 2) / (gamma - 1)) * ((Bz**2.0) / 2.0)
varvals = [0.0 for var in vars]
varvals[vars.index('h')] = rho.si

# Get dimensions of process distribution and fiddle some things
procs_index0 = outname.find('_np')+3
print outname
print [outname[i:i+2] for i in range(procs_index0, procs_index0+(2*n_dims), 2)]
nprocs = [1, 1, 1] #[int(outname[i:i+2]) for i in range(procs_index0, procs_index0+(2*n_dims), 2)]
domain_size = [int(full_domain_size[i]/nprocs[i])+4 for i in range(len(full_domain_size))]
mincoords = product(*tuple([np.arange(full_mincoords[dim], full_maxcoords[dim], (full_maxcoords[dim] - full_mincoords[dim])/p) for dim, p in enumerate(nprocs)]))
mincoords = np.array([i for i in mincoords])
maxcoords = mincoords + np.array([(full_maxcoords[dim] - full_mincoords[dim])/p for dim, p in enumerate(nprocs)])

procid = 0
for min, max in zip(mincoords, maxcoords):
    # Define file preamble
    header = header0 + ' {: 6} {: .5E} {: 1} {: 1} {: 1}\n'.format(0, 0.0, n_dims, len(eqpars), len(vars))
    for x in domain_size: header += ' {}'.format(x)
    header += '\n'
    for x in eqparvals: header += ' {: .5E}'.format(x)
    header += '\n'
    for x in dims + vars + eqpars: header += '{} '.format(x)

    # Arrange values in array for output
    outdata = np.zeros(shape=(prod(domain_size), len(dims + vars)))
    coords = product(*tuple([np.arange(min[i], max[i], (max[i]-min[i])/domain_size[i]) for i in range(len(dims))]))
    coords = np.array([i for i in coords])
    for i, dim in enumerate(dims):
        coords[:, i] += (0.5 * ((max[i]-min[i]) / domain_size[i]))
    ##### This next bit may or may not be a great big hack #####
    coords = coords[:, ::-1]
    ##### <\hack>
    outdata[:, :n_dims] = coords
    for i, val in enumerate(varvals):
        outdata[:, n_dims+i] = val
    """
     Manually adjust values as needed
    """
    # Set initial momentum
    r = np.sqrt((outdata[:, :np.max([2, n_dims])]**2.0).sum(axis=1))
    rho = outdata[:, n_dims + vars.index('h')] * u.kg / u.m**3
    """exponent = ((outdata[:, :n_dims] / delta)**2.0).sum(axis=1)
    vx = ((outdata[:, 1] / r) * np.exp(-exponent)) * amp
    vy = -((outdata[:, 0] / r) * np.exp(-exponent)) * amp"""
    phi = np.arctan2(outdata[:, 1], outdata[:, 0])
    vx = (vr * np.cos(phi)) + (vphi * np.sin(phi)).to(u.m/u.s)
    vy = (vr * np.sin(phi)) - (vphi * np.cos(phi)).to(u.m/u.s)
    outdata[:, n_dims + vars.index('m1')] = (vx * rho).si
    outdata[:, n_dims + vars.index('m2')] = (vy * rho).si
    print vx.min(), vx.max()
    print vy.min(), vy.max()
    print vx.unit.si, (vx * rho).unit.si

    # Set inital magnetic field configuration
    y = outdata[:, 1]
    # No magnetic field
    Bz = (0 * u.gauss) / np.sqrt(mu0)
    # Uniform magnetic field
    #Bz = Bz0
    # Uniform B-field inside cylinder and another uniform B-field outside
    #Bz = np.ones(domain_size) * Bz1
    #Bz[r < 0.25e6] = Bz0
    """dB = Bz1 - Bz0
    #dy = (y - full_mincoords[1]) / (full_maxcoords[1] - full_mincoords[1])
    dr = (r - r.min()) / (r.max() - r.min())
    print dr.min(), dr.max()
    print dr[0], dr[-1]
    Bz = (dB * dr) + Bz0"""
    outdata[:, n_dims + vars.index('b3')] = Bz.si

    # Set initial energy
    m1ind = n_dims + vars.index('m1')
    mvec = outdata[:, m1ind:m1ind + n_dims] * u.kg / u.m**2 / u.s
    kinE = (mvec**2.0).sum(axis=1) / rho
    magE = ((gamma - 2) / (gamma - 1)) * ((Bz**2.0) / 2.0)
    print (gasE + kinE + magE).unit.si
    outdata[:, n_dims + vars.index('e')] = (gasE + kinE + magE).si

    # Output ini info
    #ext = '.ini'#'_{:03}.ini'.format(procid)
    #np.savetxt(outname.replace('.ini', ext), outdata, header=header, comments="")#, fmt='% .10E')
    np.savetxt(outname, outdata, header=header, comments="")#, fmt='% .10E')
    procid += 1
