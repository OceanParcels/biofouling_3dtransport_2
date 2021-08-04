import numpy as np
import xarray as xr
import itertools
from scipy.signal import find_peaks
from argparse import ArgumentParser

p = ArgumentParser()
p.add_argument('-region', choices=('NPSG','EqPac','SO'), action = "store", dest = "region", help ='region where particles released')
args = p.parse_args()
region = args.region

nr_of_trajectories = 100

datadir = '/data/oceanparcels/output_data/data_Delphine/' #data_Reint/'
regions = {'EqPac': 'Equatorial Pacific',
           'GPGP': 'Great Pacific Garbage Patch',
           'SO': 'Southern Ocean',
           'NPSG': 'North Pacific Subtropical Gyre'}

mortality = 0.39              # [d-1]
runtime = 458 #822 #                # [days]
dt = 60                       # [seconds]
outputdt = 12                  # [hours]
death = 'MEDUSA'
grazing = 'full'
mixing = 'markov_0_'+death #KPP_ceiling_tides_
diss = 0.006
rho_pl = 920
rho_bf = 1170 #1388
rho_fr= 1800
sizebinedges = [1e-3, 1e-4, 1e-5]
res = '1x1'
proc = 'bfnoadv'
season = 'Oct'
season_string = {'Jan':'January - July', 'MAM':'March - September', 'JJA':'June - December', 'SON':'September - March','DJF':'December - June', 'Oct': 'October -'}

filename = datadir+'regional_'+region+'_'+proc+'_'+season+'_2003_'+res+'res_'+mixing+'_'+str(rho_bf)+'rhobf_'+str(rho_pl)+'rhopl_'+str(runtime)+'days_'+str(dt)+'dtsecs_'+str(outputdt)+'hrsoutdt.nc'
ds = xr.open_dataset(filename)

radii = np.unique(ds['r_pl'])

split_ds = list(ds.groupby('r_pl'))

oscillations = np.ones((len(split_ds),15000,1700))*-1
osc_stats = np.zeros((len(split_ds),7))

maxrange = len(ds['obs'])*nr_of_trajectories

for r in range(len(split_ds)):
    timeseries = split_ds[r][1]['z'].values.flatten()[:maxrange]
    mld = split_ds[r][1]['mld'].values.flatten()[:maxrange]
    mld_bool = timeseries>mld #> 1.15

    osc_ids = []
    for k, g in itertools.groupby(enumerate(mld_bool), lambda x:x[1]):
        if k == True:
            ind,bool = list(zip(*g))
            if len(ind)>1:
                osc_ids.append(ind[0::len(ind)-1])
    
    for i,start_end in enumerate(osc_ids):
        start, end = start_end[0], start_end[1]
        if end%len(ds['obs']) != 0 and end != maxrange-1:
            oscillations[r,i,:(end-start)] = timeseries[start:end]

    osc_stats[r, 0] = np.diff(osc_ids).mean() / (24 / outputdt)
    osc_stats[r, 1] = np.diff(osc_ids).max() / (24 / outputdt)
    osc_stats[r, 2] = np.max(oscillations[r])
    osc_stats[r, 3] = np.mean(np.max(oscillations[r], axis=1))
    osc_stats[r, 4] = len(osc_ids)
    osc_stats[r, 5] = np.percentile(np.diff(osc_ids), 90) / (24 / outputdt)
    osc_stats[r, 6] = np.percentile(np.diff(osc_ids), 99) / (24 / outputdt)
oscillations[oscillations == -1] = np.nan

np.save('/data/oceanparcels/output_data/data_Delphine/'+region+'_oscillations_rhobf'+str(rho_bf)+'_rhopl'+str(rho_pl), oscillations)
np.save('/data/oceanparcels/output_data/data_Delphine/'+region+'_oscillation_stats_rhobf'+str(rho_bf)+'_rhopl'+str(rho_pl), osc_stats)
