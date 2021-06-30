

region = 'EqPac'
mortality = 0.39              # [d-1]
runtime = 458                 # [days]
dt = 60                       # [seconds]
outputdt = 12                  # [hours]
death = 'MEDUSA'
grazing = 'full'
mixing = 'markov_0_KPP_ceiling_tides_'+death
diss = 0.006
rho_p = 920
rho_bf = 1388
rho_fr= 1800
sizebinedges = [1e-3, 1e-4, 1e-5]
res = '1x1'
proc = 'bfnoadv'
season = 'Oct'
season_string = {'Jan':'January - July', 'MAM':'March - September', 'JJA':'June - December', 'SON':'September - March','DJF':'December - June', 'Oct': 'October -'}

filename = datadir+'regional_'+region+'_'+proc+'_'+season+'_2003_'+res+'res_'+mixing+'_'+str(rho_bf)+'rhobf_'+str(runtime)+'days_'+str(dt)+'dtsecs_'+str(outputdt)+'hrsoutdt.nc'
ds = xr.open_dataset(filename)

radii = np.unique(ds['r_pl'])

split_ds = list(ds.groupby('r_pl'))

r_pl_cs = plt.get_cmap('RdPu_r', len(split_ds))

r_pl_list = r_pl_cs(np.linspace(0.,1.,len(split_ds)))

oscillations = np.ones((len(split_ds),30000,800))*-1
osc_stats = np.zeros((len(split_ds),7))

for r in range(len(split_ds)):
    timeseries = split_ds[r][1]['z'].values.flatten()
    mld = split_ds[r][1]['mld'].values.flatten()
    mld_bool = mld > 1

    osc_ids = []  # Start and end ids for single oscillations
    for k, g in itertools.groupby(enumerate(mld_bool), lambda x: x[1]):  # Groupby slices where mld > 1 / mld < 1
        if k == True:  # Only select the slices where mld > 1 -> mld_bool == True
            ind, bool = list(zip(*g))  # retrieve the indices for this slice
            if len(ind) > 1:  # Only store an oscillation if it is longer than 1 timestep
                minima = find_peaks(-timeseries[np.array(ind)[:-1]], height=(-50, 0), distance=10, prominence=2)
                #                 minima = argrelextrema(timeseries[np.array(ind)[:-1]], np.less, order=20)       # Check whether there are local minima in the oscillation that should be separated
                #             print(minima)
                if len(minima[0]) > 0:
                    osc_ids.append((ind[0], ind[0] + minima[0][0]))
                    for i in range(len(minima[0]) - 1):
                        osc_ids.append((ind[0] + minima[0][i], ind[0] + minima[0][i + 1]))
                    if len(minima[0]) > 1:
                        osc_ids.append((ind[0] + minima[0][-1], ind[-1]))
                else:
                    osc_ids.append(ind[0::len(ind) - 1])

    for i, start_end in enumerate(osc_ids):
        start, end = start_end[0], start_end[1]
        #         if timeseries[end]<50:
        oscillations[r, i, :(end - start)] = timeseries[start:end]
    osc_stats[r, 0] = np.diff(osc_ids).mean() / (24 / outputdt)
    osc_stats[r, 1] = np.diff(osc_ids).max() / (24 / outputdt)
    osc_stats[r, 2] = np.max(oscillations[r])
    osc_stats[r, 3] = np.mean(np.max(oscillations[r], axis=1))
    osc_stats[r, 4] = len(osc_ids)
    osc_stats[r, 5] = np.percentile(np.diff(osc_ids), 90) / (24 / outputdt)
    osc_stats[r, 6] = np.percentile(np.diff(osc_ids), 10) / (24 / outputdt)
oscillations[oscillations == -1] = np.nan

np.save(region+'_oscillations', oscillations)
np.save(region+'_oscillation_stats', osc_stats)