"""
Authors: Delphine Lobelle, Reint Fischer

Executable python script to simulate regional biofouling particles with parameterized wind and tidal mixing.
"""

from parcels import FieldSet, ParticleSet, AdvectionRK4_3D, ErrorCode, ParticleFile, Field
from parcels.application_kernels.TEOSseawaterdensity import PolyTEOS10_bsq
from kernels import plastic_particle, MEDUSA_biofouling, MEDUSA_detritus, markov_0_mixing, profiles, AdvectionRK4_1D
from utils import delete_particle, delete_particle_interp, periodicBC, getclosest_ij, uniform_release
from datetime import timedelta as delta
import numpy as np
from glob import glob
import xarray as xr
from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore")

#------ Choose ------:
simdays = 458
secsdt = 60
hrsoutdt = 12

if __name__ == "__main__":     
    p = ArgumentParser(description="""choose starting month and year""")
    p.add_argument('-mon', choices = ('01','12','03','06','09','10'), action="store", dest="mon", 
                   help='start month for the run')
    p.add_argument('-yr', choices = ('2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010'), action="store", dest="yr",
                   help='start year for the run')
    p.add_argument('-region', choices = ('NPSG','EqPac','SO'), action = "store", dest = "region", help ='region where particles released')
    p.add_argument('-mixing', choices = ('no', 'markov_0'), action = "store", dest = 'mixing', help='Type of random vertical mixing. "no" is none, "fixed" is mld between 0.2 and -0.2 m/s')
    p.add_argument('-biofouling', choices=('MEDUSA_detritus', 'MEDUSA'), action='store', dest = 'biofouling')
    p.add_argument('-rhobf', choices=('1388', '1170'), action='store', dest = 'rhobf')
    p.add_argument('-rhopl', choices=('30', '920'), action='store', dest = 'rhopl')
    p.add_argument('-system', choices=('gemini', 'cartesius', 'lorenz'), action='store', dest = 'system', help='"gemini", "lorenz" or "cartesius"')
    p.add_argument('-no_advection', choices =('True','False'), action="store", dest="no_advection", help='True if removing advection_RK43D kernel')
 
    args = p.parse_args()
    mon = args.mon
    yr = args.yr
    region = args.region
    mixing = args.mixing
    biofouling = args.biofouling
    rhopl = args.rhopl
    rhobf = args.rhobf
    system = args.system
    no_advection = str(args.no_advection)
 
    """ Define simulation domain """
    #------ Fieldset grid  ------
    if region == 'NPSG' and no_advection == 'False':
        minlat = -30 
        maxlat = 70 
        minlon = 110
        maxlon = -100
    elif region == 'NPSG':
        minlat = 22
        maxlat = 35
        minlon = -144
        maxlon = -133
    elif region == 'EqPac' and no_advection == 'False':
        minlat = -40 
        maxlat = 40
        minlon = 100
        maxlon = -90
    elif region == 'EqPac':
        minlat = -5
        maxlat = 5
        minlon = -149
        maxlon = -138
    elif region == 'SO' and no_advection == 'False':
        minlat = -75
        maxlat = -15
        minlon = 90
        maxlon = 60
    elif region == 'SO':
        minlat= -63
        maxlat = -52
        minlon = -116
        maxlon = -105

    """ Defining the fieldset""" 
    if system == 'cartesius':
        dirread = '/projects/0/topios/hydrodynamic_data/NEMO-MEDUSA/ORCA0083-N006/means/'
        dirread_bgc = '/projects/0/topios/hydrodynamic_data/NEMO-MEDUSA_BGC/ORCA0083-N006/means/'  
        dirread_mesh = '/projects/0/topios/hydrodynamic_data/NEMO-MEDUSA/ORCA0083-N006/domain/'
    elif system == 'gemini':
        dirread = '/data/oceanparcels/input_data/NEMO-MEDUSA/ORCA0083-N006/means/'
        dirread_bgc = '/data/oceanparcels/input_data/NEMO-MEDUSA_BGC/ORCA0083-N006/means/'
        dirread_mesh = '/data/oceanparcels/input_data/NEMO-MEDUSA/ORCA0083-N006/domain/'
    elif system == 'lorenz':
        dirread = '/storage/shared/oceanparcels/input_data/NEMO-MEDUSA/ORCA0083-N006/means/'
        dirread_bgc = '/storage/shared/oceanparcels/input_data/NEMO-MEDUSA_BGC/ORCA0083-N006/means/'
        dirread_mesh = '/storage/shared/oceanparcels/input_data/NEMO-MEDUSA/ORCA0083-N006/domain/'

    startdate = np.datetime64(f'{yr}-{mon}-01')
    enddate = startdate + np.timedelta64(simdays+5,'D')

    startyear = int(str(startdate)[:4])
    endyear = int(str(enddate)[:4])

    if mon == '01':
        yr0 = str(int(yr) - 1)
        ufiles = (sorted(glob(dirread + 'ORCA0083-N06_' + yr0 + '*d05U.nc'))+sorted(glob(dirread + 'ORCA0083-N06_' + yr + '*d05U.nc')))
        vfiles = (sorted(glob(dirread + 'ORCA0083-N06_' + yr0 + '*d05V.nc'))+sorted(glob(dirread + 'ORCA0083-N06_' + yr + '*d05V.nc')))
        wfiles = (sorted(glob(dirread + 'ORCA0083-N06_' + yr0 + '*d05W.nc'))+sorted(glob(dirread + 'ORCA0083-N06_' + yr + '*d05W.nc')))
        pfiles = (sorted(glob(dirread_bgc + 'ORCA0083-N06_' + yr0 + '*d05P.nc'))+sorted(glob(dirread_bgc + 'ORCA0083-N06_' + yr + '*d05P.nc')))
        ppfiles = (sorted(glob(dirread_bgc + 'ORCA0083-N06_' + yr0 + '*d05D.nc'))+sorted(glob(dirread_bgc + 'ORCA0083-N06_' + yr + '*d05D.nc')))
        tsfiles = (sorted(glob(dirread + 'ORCA0083-N06_' + yr0 + '*d05T.nc'))+sorted(glob(dirread + 'ORCA0083-N06_' + yr + '*d05T.nc')))
    else:
        ufiles = (sorted(glob(dirread + 'ORCA0083-N06_' + yr + '*d05U.nc')))
        vfiles = (sorted(glob(dirread + 'ORCA0083-N06_' + yr + '*d05V.nc')))
        wfiles = (sorted(glob(dirread + 'ORCA0083-N06_' + yr + '*d05W.nc')))
        pfiles = (sorted(glob(dirread_bgc + 'ORCA0083-N06_' + yr + '*d05P.nc')))
        ppfiles = (sorted(glob(dirread_bgc + 'ORCA0083-N06_' + yr + '*d05D.nc')))
        tsfiles = (sorted(glob(dirread + 'ORCA0083-N06_' + yr + '*d05T.nc')))

    for i in range(endyear-startyear):
        newyr = str(int(yr)+i+1)
        ufiles += (sorted(glob(dirread + 'ORCA0083-N06_' + newyr + '*d05U.nc')))
        vfiles += (sorted(glob(dirread + 'ORCA0083-N06_' + newyr + '*d05V.nc')))
        wfiles += (sorted(glob(dirread + 'ORCA0083-N06_' + newyr + '*d05W.nc')))
        pfiles += (sorted(glob(dirread_bgc + 'ORCA0083-N06_' + newyr + '*d05P.nc')))
        ppfiles += (sorted(glob(dirread_bgc + 'ORCA0083-N06_' + newyr + '*d05D.nc')))
        tsfiles += (sorted(glob(dirread + 'ORCA0083-N06_' + newyr + '*d05T.nc')))

        
    mesh_mask = dirread_mesh+'coordinates.nc'

    filenames = {'U': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': ufiles}, #'depth': wfiles,
                 'V': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': vfiles},
                 'W': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': wfiles},
                 'd_phy': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': pfiles},
                 'nd_phy': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': pfiles},
                 'tpp3': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': ppfiles},
                 'cons_temperature': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': tsfiles},
                 'abs_salinity': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': tsfiles},
                 'mldr': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': tsfiles},
                 'taum': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': tsfiles},
                 'w_10': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': tsfiles},
                 'euph_z': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': ppfiles},
                 'mic_zoo': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': pfiles},
                 'mes_zoo': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': pfiles},   
                 'detr': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': pfiles},
                 'Di_Si': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': pfiles}}

    variables = {'U': 'uo',
                 'V': 'vo',
                 'W': 'wo', 
                 'd_phy': 'PHD',                # units: mmolN/m3
                 'nd_phy': 'PHN',               # units: mmolN/m3
                 'tpp3': 'TPP3',                # units: mmolN/m3/d 
                 'cons_temperature': 'potemp',
                 'abs_salinity': 'salin',
                 'mldr': 'mldr10_1',
                 'taum': 'taum',
                 'w_10': 'sowindsp',
                 'euph_z': 'MED_XZE',           # units: m
                 'mic_zoo': 'ZMI',              # units: mmolN/m3
                 'mes_zoo': 'ZME',              # units: mmolN/m3
                 'detr': 'DET',                 # units: mmolN/m3
                 'Di_Si': 'PDS'}                # units: mmolSi/m3

    dimensions = {'U': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'}, #time_centered
                  'V': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'},
                  'W': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'},
                  'd_phy': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw','time': 'time_counter'},
                  'nd_phy': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw','time': 'time_counter'},
                  'tpp3': {'lon': 'glamf', 'lat': 'gphif','depth': 'depthw', 'time': 'time_counter'},
                  'cons_temperature': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw','time': 'time_counter'},
                  'abs_salinity': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw','time': 'time_counter'},
                  'mldr': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'},
                  'taum': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'},
                  'w_10': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'},
                  'euph_z': {'lon':'glamf', 'lat':'gphif', 'time': 'time_counter'},
                  'mic_zoo': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw','time': 'time_counter'},
                  'mes_zoo': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw','time': 'time_counter'},
                  'detr': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw','time': 'time_counter'},
                  'Di_Si': {'lon':'glamf', 'lat':'gphif', 'time': 'time_counter'}}

    initialgrid_mask = dirread+'ORCA0083-N06_20040105d05U.nc'
    mask = xr.open_dataset(initialgrid_mask, decode_times=False)
    Lat, Lon = mask.variables['nav_lat'], mask.variables['nav_lon']
    latvals = Lat[:]; lonvals = Lon[:] # extract lat/lon values to numpy arrays
                                                                                               
    iy_min, ix_min = getclosest_ij(latvals, lonvals, minlat, minlon)
    iy_max, ix_max = getclosest_ij(latvals, lonvals, maxlat, maxlon)

    indices = {'lat': range(iy_min, iy_max), 'lon': range(ix_min, ix_max)} #depth : range(0,2000)

    chs = {'U': {'time': ('time_counter', 1), 'depth': ('depthu', 25), 'lat': ('y', 200), 'lon': ('x', 200)},
           'V': {'time': ('time_counter', 1), 'depth': ('depthv', 25), 'lat': ('y', 200), 'lon': ('x', 200)},
           'W': {'time': ('time_counter', 1), 'depth': ('depthw', 25), 'lat': ('y', 200), 'lon': ('x', 200)},
           'd_phy': {'time': ('time_counter', 1), 'depth': ('deptht', 25), 'lat': ('y', 200), 'lon': ('x', 200)},
           'nd_phy': {'time': ('time_counter', 1), 'depth': ('deptht', 25), 'lat': ('y', 200), 'lon': ('x', 200)},
           'tpp3': {'time': ('time_counter', 1), 'depth': ('deptht', 25), 'lat': ('y', 200), 'lon': ('x', 200)},
           'cons_temperature': {'time': ('time_counter', 1), 'depth': ('deptht', 25), 'lat': ('y', 200), 'lon': ('x', 200)},
           'abs_salinity': {'time': ('time_counter', 1), 'depth': ('deptht', 25), 'lat': ('y', 200), 'lon': ('x', 200)},
           'mldr': {'time': ('time_counter', 1), 'depth': ('deptht', 25), 'lat': ('y', 200), 'lon': ('x', 200)},
           'taum': {'time': ('time_counter', 1), 'depth': ('deptht', 25), 'lat': ('y', 200), 'lon': ('x', 200)},
           'w_10': {'time': ('time_counter', 1), 'depth': ('deptht', 25), 'lat': ('y', 200), 'lon': ('x', 200)},
           'euph_z': {'time': ('time_counter', 1), 'depth': ('deptht', 25), 'lat':('y', 200), 'lon': ('x', 200)},
           'mic_zoo': {'time': ('time_counter', 1), 'depth': ('deptht', 25), 'lat':('y', 200), 'lon': ('x', 200)},
           'mes_zoo': {'time': ('time_counter', 1), 'depth': ('deptht', 25), 'lat':('y', 200), 'lon': ('x', 200)},
           'detr': {'time': ('time_counter', 1), 'depth': ('deptht', 25), 'lat':('y', 200), 'lon': ('x', 200)},
           'Di_Si': {'time': ('time_counter', 1), 'depth': ('deptht', 25), 'lat':('y', 200), 'lon': ('x', 200)}}
    
    fieldset = FieldSet.from_nemo(filenames, variables, dimensions, allow_time_extrapolation=False, indices = indices, chunksize=chs)

    variable = ('Kz', 'TIDAL_Kz')
    dimension = {'lon': 'Longitude', 'lat': 'Latitude', 'depth':'Depth_midpoint'}
    if system == 'gemini':
        Kz_field = Field.from_netcdf('/scratch/rfischer/Kooi_data/data_input/Kz.nc', variable, dimension)
    elif system == 'cartesius':
        Kz_field = Field.from_netcdf('/home/dlobelle/biofouling_3dtransport_2/Preprocessing/Kz.nc', variable, dimension)
    elif system == 'lorenz':
        Kz_field = Field.from_netcdf('Kz.nc', variable, dimension)
    fieldset.add_field(Kz_field)

    variabled = ('dKzdz', 'TIDAL_dKz')
    if system == 'gemini':
        dKz_field = Field.from_netcdf('/scratch/rfischer/Kooi_data/data_input/Kz.nc', variabled, dimension)
    elif system == 'cartesius':
        dKz_field = Field.from_netcdf('/home/dlobelle/biofouling_3dtransport_2/Preprocessing/Kz.nc', variabled, dimension)
    elif system == 'lorenz':
        dKz_field = Field.from_netcdf('Kz.nc', variabled, dimension)
    fieldset.add_field(dKz_field)

    # ------ Defining constants ------
    fieldset.add_constant('Gr_a', 0.39 / 86400.)
    fieldset.add_constant('collision_eff', 1.)
    fieldset.add_constant('K', 1.0306E-13 / (86400. ** 2.))  # Boltzmann constant [m2 kg d-2 K-1] now [s-2] (=1.3804E-23)
    fieldset.add_constant('Rho_bf', int(rhobf)) #1388.)      # density of biofilm [g m-3]
    fieldset.add_constant('Rho_fr', 1800.)                   # density of frustule [g m-3] median value from Miklasz & Denny 2010
    fieldset.add_constant('Rho_cy', 1065.)                   # density of cytoplasm [g m-3] median value from Miklasz & Denny 2010
    fieldset.add_constant('V_a', 2.0E-16)                    # Volume of 1 algal cell [m-3]
    fieldset.add_constant('R20', 0.1 / 86400.)               # respiration rate, now [s-1]
    fieldset.add_constant('Q10', 2.)                         # temperature coefficient respiration [-]
    fieldset.add_constant('Gamma', 1.728E5 / 86400.)         # shear [d-1], now [s-1]
    fieldset.add_constant('Wt_N', 14.007)                    # atomic weight of nitrogen
    fieldset.add_constant('G', 7.32e10/(86400.**2.))
    

    # ------ MEDUSA constants ------
    # Derived from MEDUSA 2.0 Yool et al. 2013
    fieldset.add_constant('D1', 0.33)                        # Fast detritus fraction of diatom losses
    fieldset.add_constant('D2', 1.)
    fieldset.add_constant('D3', 0.8)
    fieldset.add_constant('mu1', 0.02/86400.)                # Linear diatom loss rate [s-1]
    fieldset.add_constant('mu2', 0.1/86400.)                 # Non-Linear maximum diatom loss rate [s-1]
    fieldset.add_constant('kPd', 0.5)                        # Diatom loss half-saturation constant [mmol N m-3]
    fieldset.add_constant('Wt_Si', 28.0855)                  # Si atomic weight
    fieldset.add_constant('R_N_Si_min', 0.2)                 # Minimum N:Si ratio
    fieldset.add_constant('R_N_Si_max', 5.)                  # Maximum N:Si ratio
    fieldset.add_constant('Diss', 0.006/86400.)              # Dissolution rate [d-1] -> [s-1]
    fieldset.add_constant('Gm', 0.5/86400.)                  # Maximum zooplankton grazing rate [s-1]
    fieldset.add_constant('km', 0.3)                         # Zooplankton grazing half-saturation constant [mmol N m-3]
    fieldset.add_constant('pmPn', 0.15)                      # Mesozooplankton grazing preference for non-diatoms
    fieldset.add_constant('pmPd', 0.35)                      # Mesozooplankton grazing preference for diatoms
    fieldset.add_constant('pmZmu', 0.35)                     # Mesozooplankton grazing preference for microzooplankton
    fieldset.add_constant('pmD', 0.15)                       # Mesozooplankton grazing preference for detritus

    if mixing == 'markov_0':
        fieldset.add_constant('Vk', 0.4)
        fieldset.add_constant('Phi', 0.9)
        fieldset.add_constant('Rho_a', 1.22)
        fieldset.add_constant('Wave_age', 35)

    """ Defining the particle set """   
    n_res = 10
    n_locs = n_res**2
    n_sizebins = 25
    n_particles_per_bin = 4

    if region == 'NPSG':
        lat_release0 = np.tile(np.linspace(23,32,n_res),[n_res,1])
        lon_release0 = np.tile(np.linspace(-143,-134,n_res),[n_res,1])

        lat_release = np.tile(lat_release0.T, [n_sizebins*n_particles_per_bin,1,1]).T
        lon_release = np.tile(lon_release0, [n_sizebins*n_particles_per_bin,1,1]).T
    elif region == 'EqPac':
        lat_release0 = np.tile(np.linspace(-4.5,4.5,n_res),[n_res,1])
        lon_release0 = np.tile(np.linspace(-148,-139,n_res),[n_res,1])

        lat_release = np.tile(lat_release0.T, [n_sizebins*n_particles_per_bin,1,1]).T
        lon_release = np.tile(lon_release0, [n_sizebins*n_particles_per_bin,1,1]).T
    elif region == 'SO':
        lat_release0  = np.tile(np.linspace(-62,-53,n_res),[n_res,1])
        lon_release0 = np.tile(np.linspace(-115,-106,n_res),[n_res,1])

        lat_release = np.tile(lat_release0.T, [n_sizebins*n_particles_per_bin,1,1]).T
        lon_release = np.tile(lon_release0, [n_sizebins*n_particles_per_bin,1,1]).T

    z_release = np.tile(0.6,[n_res,n_res, n_sizebins*n_particles_per_bin])
    res = '1x1'

    rho_pls = np.tile(int(rhopl), [n_res, n_res, n_sizebins*n_particles_per_bin])
    rho_bfs = np.tile(int(rhobf), [n_res, n_res, n_sizebins*n_particles_per_bin])
    r_pls = uniform_release(n_locs, n_particles_per_bin, n_sizebins)

    pset = ParticleSet.from_list(fieldset=fieldset,         # the fields on which the particles are advected
                                 pclass=plastic_particle,   # the type of particles (JITParticle or ScipyParticle)
                                 lon= lon_release, #-160.,  # a vector of release longitudes 
                                 lat= lat_release, #36., 
                                 time = startdate,
                                 rho_bf = rho_bfs,
                                 depth = z_release,
                                 r_pl = r_pls,
                                 rho_pl = rho_pls,
                                 r_tot = r_pls,
                                 rho_tot = rho_pls)


    """ Selecting kernels + Execution"""
    if mon=='12':
        s = 'DJF'
    elif mon=='03':
        s = 'MAM'
    elif mon=='06':
        s = 'JJA'
    elif mon=='09':
        s = 'SON'
    elif mon=='10':
        s = 'Oct'
    elif mon=='01':
        s = 'Jan'

    kernels = pset.Kernel(PolyTEOS10_bsq) + pset.Kernel(profiles)

    if no_advection == 'True':
        proc = 'bfnoadv'
        kernels += pset.Kernel(AdvectionRK4_1D)
    elif no_advection == 'False':
        proc = 'bfadv'
        kernels += pset.Kernel(AdvectionRK4_3D)

    kernels += pset.Kernel(periodicBC)

    if mixing == 'markov_0':
        kernels += pset.Kernel(markov_0_mixing)

    if biofouling == 'MEDUSA':
        kernels += pset.Kernel(MEDUSA_biofouling)
    elif biofouling == 'MEDUSA_detritus':
        kernels += pset.Kernel(MEDUSA_detritus)
        
    if system == 'cartesius':
        outfile = '/home/dlobelle/biofouling_3dtransport_2/Simulation/Sim_output/regional_'+region+'_'+proc+'_'+s+'_'+yr+'_'+res+'res_'+mixing+'_'+biofouling+'_'+str(int(fieldset.Rho_bf))+'rhobf_'+str(int(rhopl))+'rhopl_'+str(round(simdays,2))+'days_'+str(secsdt)+'dtsecs_'+str(round(hrsoutdt,2))+'hrsoutdt' 
    elif system == 'gemini':
        outfile = '/data/oceanparcels/output_data/data_Delphine/regional_'+region+'_'+proc+'_'+s+'_'+yr+'_'+res+'res_'+mixing+'_'+biofouling+'_'+str(int(fieldset.Rho_bf))+'rhobf_'+str(int(rhopl))+'rhopl_'+str(round(simdays,2))+'days_'+str(secsdt)+'dtsecs_'+str(round(hrsoutdt,2))+'hrsoutdt' 
    elif system == 'lorenz':
        outfile = 'regional_'+region+'_'+proc+'_'+s+'_'+yr+'_'+res+'res_'+mixing+'_'+biofouling+'_'+str(int(fieldset.Rho_bf))+'rhobf_'+str(int(rhopl))+'rhopl_'+str(round(simdays,2))+'days_'+str(secsdt)+'dtsecs_'+str(round(hrsoutdt,2))+'hrsoutdt'

    pfile= ParticleFile(outfile, pset, outputdt=delta(hours = hrsoutdt))

    pset.execute(kernels, runtime=delta(days=simdays), dt=delta(seconds = secsdt), output_file=pfile, verbose_progress=True, recovery={ErrorCode.ErrorOutOfBounds: delete_particle, ErrorCode.ErrorInterpolation: delete_particle_interp})

    pfile.close()

    print('Execution finished')