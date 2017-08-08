import os
import astropy.units as u
import numpy as np
#from pysac.mhs_atmosphere.parameters.model_pars import drew_paper1 as model_pars
from pysac.mhs_atmosphere.parameters.model_pars import drew_model as model_pars
#from pysac.mhs_atmosphere.parameters.model_pars import mfe_setup as model_pars
import pysac.mhs_atmosphere as atm
import yt
from astropy.modeling import models, fitting
import sys

# Cheeky Reset to Photosphere
model_pars['xyz'][4] = 0*u.Mm
#==============================================================================
# Build the MFE flux tube model using pysac
#==============================================================================
# model setup
scales, physical_constants = atm.units_const.get_parameters()
option_pars = atm.options.set_options(model_pars, False, l_gdf=True)
coords = atm.model_pars.get_coords(model_pars['Nxyz'], u.Quantity(model_pars['xyz']))

#interpolate the hs 1D profiles from empirical data source[s]
empirical_data = atm.hs_atmosphere.read_VAL3c_MTW(mu=physical_constants['mu'])
table = atm.hs_atmosphere.interpolate_atmosphere(empirical_data, coords['Zext'])
table['rho'] += table['rho'].min()*3.6

# calculate 1d hydrostatic balance from empirical density profile
# the hs pressure balance is enhanced by pressure equivalent to the
# residual mean coronal magnetic pressure contribution once the magnetic
# field has been applied
magp_meanz = np.ones(len(coords['Z'])) * u.one
magp_meanz *= model_pars['pBplus']**2/(2*physical_constants['mu0'])

# Make the vertical profile 3D
pressure_z, rho_z, Rgas_z = atm.hs_atmosphere.vertical_profile(coords['Z'], table, magp_meanz, physical_constants, coords['dz'])

# Generate 3D coordinate arrays
x, y, z = u.Quantity(np.mgrid[coords['xmin']:coords['xmax']:1j*model_pars['Nxyz'][0],
                              coords['ymin']:coords['ymax']:1j*model_pars['Nxyz'][1],
                              coords['zmin']:coords['zmax']:1j*model_pars['Nxyz'][2]], unit=coords['xmin'].unit)

# Get default MFE flux tube parameters out of pysac
xi, yi, Si = atm.flux_tubes.get_flux_tubes(model_pars, coords, option_pars)

# Generate the 3D magnetic field
allmag = atm.flux_tubes.construct_magnetic_field(x, y, z, xi[0], yi[0], Si[0], model_pars, option_pars, physical_constants, scales)
pressure_m, rho_m, Bx, By ,Bz, Btensx, Btensy = allmag

# local proc 3D mhs arrays
pressure, rho = atm.mhs_3D.mhs_3D_profile(z, pressure_z, rho_z, pressure_m, rho_m)
magp = (Bx**2 + By**2 + Bz**2) / (2.*physical_constants['mu0'])
energy = atm.mhs_3D.get_internal_energy(pressure, magp, physical_constants)

# Add derived Fields
def magnetic_field_strength(field, data):
    return np.sqrt(data["magnetic_field_x"]**2 + data["magnetic_field_y"]**2 + data["magnetic_field_z"]**2)
yt.add_field(("gas","magnetic_field_strength"), function=magnetic_field_strength, units=yt.units.T.units)

def alfven_speed(field, data):
    return np.sqrt(2.*data['magnetic_pressure']/data['density'])
yt.add_field(("gas","alfven_speed"), function=alfven_speed, units=(yt.units.m/yt.units.s).units)

bbox = u.Quantity([u.Quantity([coords['xmin'], coords['xmax']]),
                   u.Quantity([coords['ymin'], coords['ymax']]),
                   u.Quantity([coords['zmin'], coords['zmax']])]).to(u.m).value

# Now build a yt DataSet with the generated data:
data = {'magnetic_field_x':yt.YTQuantity.from_astropy(Bx.si),#decompose()),
        'magnetic_field_y':yt.YTQuantity.from_astropy(By.si),#decompose()),
        'magnetic_field_z':yt.YTQuantity.from_astropy(Bz.si),#decompose()),
        'pressure': yt.YTQuantity.from_astropy(pressure.si),#decompose()),
        'magnetic_pressure': yt.YTQuantity.from_astropy(magp.si),#decompose()),
        'density': yt.YTQuantity.from_astropy(rho.si),#decompose()),
        'energy': yt.YTQuantity.from_astropy(energy.si)}#decompose())}

ds = yt.load_uniform_grid(data, x.shape, length_unit='m', magnetic_unit='T', mass_unit='kg', periodicity=[False]*3, bbox=bbox)

x_lin = np.linspace(coords['xmin'].value, coords['xmax'].value, model_pars['Nxyz'][0])*coords['xmin'].unit
#bmag = np.sqrt((Bx**2 + By**2 + Bz**2))[:,64,0].to(u.mT) ######
#bmag = np.sqrt((Bx**2 + By**2 + Bz**2))[:,128,0].to(u.mT) ######
print Bx.shape[1]/2
bmag = np.sqrt((Bx**2 + By**2 + Bz**2))[:,Bx.shape[1]/2,0].to(u.mT) ######

gaussian = models.Gaussian1D()
bmag_fit = fitting.LevMarLSQFitter()(gaussian, x_lin, bmag)

fwhm = 2.*np.sqrt(2*np.log(2))*bmag_fit.stddev.value
fwhm = np.abs(fwhm) * u.Mm

fwtm = 2.*np.sqrt(2*np.log(10))*bmag_fit.stddev.value
fwtm = np.abs(fwtm) * u.Mm

fwhm = fwhm.to(u.km) 
fwtm = fwtm.to(u.km)

#============================================================================
# Save data for SAC and plotting
#============================================================================
# set up data directory and file names
# may be worthwhile locating on /data if files are large
datadir = os.path.expanduser('/data/sm1ajl/mhs_atmosphere/'+model_pars['model']+'/')
filename = datadir + model_pars['model'] + option_pars['suffix']
print '\n\n====', datadir, filename, '====\n\n'
if not os.path.exists(datadir):
    os.makedirs(datadir)
sourcefile = datadir + model_pars['model'] + '_sources' + option_pars['suffix']
aux3D = datadir + model_pars['model'] + '_3Daux' + option_pars['suffix']
aux1D = datadir + model_pars['model'] + '_1Daux' + option_pars['suffix']
print 'units of Bx =',Bx.unit
# save the variables for the initialisation of a SAC simulation
atm.mhs_snapshot.save_SACvariables(
              filename,
              rho,
              Bx,
              By,
              Bz,
              energy,
              option_pars,
              physical_constants,
              coords,
              model_pars['Nxyz'],
              np.s_[:]
             )
"""# save the balancing forces as the background source terms for SAC simulation
atm.mhs_snapshot.save_SACsources(
              sourcefile,
              Fx,
              Fy,
              option_pars,
              physical_constants,
              coords,
              model_pars['Nxyz']
             )
# save auxilliary variable and 1D profiles for plotting and analysis
Rgas = u.Quantity(np.zeros(x.shape), unit=Rgas_z.unit)
Rgas[:] = Rgas_z
temperature = pressure/rho/Rgas
if not option_pars['l_hdonly']:
    inan = np.where(magp <=1e-7*pressure.min())
    magpbeta = magp
    magpbeta[inan] = 1e-7*pressure.min()  # low pressure floor to avoid NaN
    pbeta  = pressure/magpbeta
else:
    pbeta  = magp+1.0    #dummy to avoid NaN
alfven = np.sqrt(2.*magp/rho)
if rank == 0:
    print'Alfven speed Z.min to Z.max =',\
    alfven[model_pars['Nxyz'][0]/2,model_pars['Nxyz'][1]/2, 0].decompose(),\
    alfven[model_pars['Nxyz'][0]/2,model_pars['Nxyz'][1]/2,-1].decompose()

cspeed = np.sqrt(physical_constants['gamma']*pressure/rho)
atm.mhs_snapshot.save_auxilliary3D(
              aux3D,
              pressure_m,
              rho_m,
              temperature,
              pbeta,
              alfven,
              cspeed,
              Btensx,
              Btensy,
              option_pars,
              physical_constants,
              coords,
              model_pars['Nxyz']
             )
atm.mhs_snapshot.save_auxilliary1D(
              aux1D,
              pressure_Z,
              rho_Z,
              Rgas_Z,
              option_pars,
              physical_constants,
              coords,
              model_pars['Nxyz']
             )
"""
