import pysac
import pysac.yt
import pysac.mhs_atmosphere.plot.mhs_plot as mhsp
import yt

ds1 = yt.load('/data/sm1ajl/custom_ini_files/drew_model/drew_model.gdf')
ds2 = yt.load('/data/sm1ajl/custom_ini_files/mfe_setup/mfe_setup.gdf')

profiles = {}
for i, ds in enumerate([ds1, ds2]):
    profiles = mhsp.make_1d_slices(ds, 'mag_pressure', profiles)
    profiles = mhsp.make_1d_slices(ds, 'thermal_pressure', profiles)

    tpr = profiles['thermal_pressure']['axis'] + profiles['mag_pressure']['axis']
    profiles['total_pressure'] = {'axis': tpr, 'Z': profiles['thermal_pressure']['Z']}
    tp = profiles['total_pressure']['axis']

    profiles = mhsp.make_1d_slices(ds, 'density_bg', profiles)
    profiles = mhsp.make_1d_slices(ds, 'density_pert', profiles)

    dbg = profiles['density_bg']['axis']
    ddbg = dbg[1:] - dbg[:-1]
    profiles['ddens_{}'.format(i)] = {'axis': ddbg, 'Z': profiles['density_bg']['Z'][1:]}

    mhsp.make_1d_zplot(profiles, 'density_{}'.format(i),
                       ['density_bg'],
                       figxy=(10, 8), loc_legend='upper right', ylog=False)

    pbg = profiles['thermal_pressure']['axis']
    dpbg = pbg[1:] - pbg[:-1]
    profiles['dpress_{}'.format(i)] = {'axis': dpbg, 'Z': profiles['thermal_pressure']['Z'][1:]}

    mhsp.make_1d_zplot(profiles, 'pressure_{}'.format(i),
                       ['mag_pressure'],
                       figxy=(10, 8), loc_legend='upper right', ylog=False)

    #dfrac = profiles['density_pert']['axis'] / profiles['density_bg']['axis']
    #profiles['density_frac'] = {'axis': dfrac, 'Z': profiles['density_bg']['Z']}

    profiles = mhsp.make_1d_slices(ds, 'internal_energy_bg', profiles)
    ebg = profiles['internal_energy_bg']['axis']
    debg = ebg[1:] - ebg[:-1]
    profiles['dener_{}'.format(i)] = {'axis': debg, 'Z': profiles['internal_energy_bg']['Z'][1:]}

    profiles = mhsp.make_1d_slices(ds, 'temperature', profiles)
    tbg = profiles['temperature'.format(i)]['axis']
    dtbg = tbg[1:] - tbg[:-1]
    profiles['dtemp_{}'.format(i)] = {'axis': dtbg, 'Z': profiles['temperature']['Z'][1:]}

    mhsp.make_1d_zplot(profiles, 'temp_{}'.format(i),
                       ['temperature'],
                       figxy=(10, 8), loc_legend='upper right', ylog=False)

mhsp.make_1d_zplot(profiles, 'density',
                   ['ddens_0', 'ddens_1'],
                   figxy=(10, 8), loc_legend='upper right', ylog=False)

mhsp.make_1d_zplot(profiles, 'temp',
                   ['dtemp_0', 'dtemp_1'],
                   figxy=(10, 8), loc_legend='upper right', ylog=False)

mhsp.make_1d_zplot(profiles, 'pressure',
                   #['total_pressure', 'thermal_pressure', 'mag_pressure'],
                   ['dpress_0', 'dpress_1'],
                   #['density_bg'],
                   figxy=(10, 8), loc_legend='upper right', ylog=False)
