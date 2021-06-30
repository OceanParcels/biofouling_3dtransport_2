import numpy as np
from glob import glob
import xarray as xr
from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore")

# compute climatology for one region

p = ArgumentParser()
p.add_argument('-region', choices=('NPSG','EqPac','SO'), action = "store", dest = "region", help ='region where particles released')
args = p.parse_args()
region = args.region

yr = '2004'
dirread_NEMO = '/data/oceanparcels/input_data/NEMO-MEDUSA/ORCA0083-N006/means/'
dirread_bgc_NEMO = '/data/oceanparcels/input_data/NEMO-MEDUSA_BGC/ORCA0083-N006/means/'
dirread_mesh = '/data/oceanparcels/input_data/NEMO-MEDUSA/ORCA0083-N006/domain/'

ufiles_NEMO = sorted(glob(dirread_NEMO+'ORCA0083-N06_'+yr+'*d05U.nc'))
vfiles_NEMO = sorted(glob(dirread_NEMO+'ORCA0083-N06_'+yr+'*d05V.nc'))
wfiles_NEMO = sorted(glob(dirread_NEMO+'ORCA0083-N06_'+yr+'*d05W.nc'))
pfiles_NEMO = sorted(glob(dirread_bgc_NEMO+'ORCA0083-N06_'+yr+'*d05P.nc'))
ppfiles_NEMO = sorted(glob(dirread_bgc_NEMO+'ORCA0083-N06_'+yr+'*d05D.nc'))
tsfiles_NEMO = sorted(glob(dirread_NEMO+'ORCA0083-N06_'+yr+'*d05T.nc'))
mesh_mask_NEMO = dirread_mesh+'coordinates.nc'

ds_pp_NEMO = xr.open_dataset(ppfiles_NEMO[0])
ds_p_NEMO = xr.open_dataset(pfiles_NEMO[0])
ds_ts_NEMO = xr.open_dataset(tsfiles_NEMO[0])

mesh_mask = xr.open_dataset(mesh_mask_NEMO, decode_times=False)

def getclosest_ij(lats,lons,latpt,lonpt):
    """Function to find the index of the closest point to a certain lon/lat value."""
    dist_sq = (lats-latpt)**2 + (lons-lonpt)**2                 # find squared distance of every point on grid
    minindex_flattened = dist_sq.argmin()                       # 1D index of minimum dist_sq element
    return np.unravel_index(minindex_flattened, lats.shape)     # Get 2D index for latvals and lonvals arrays from 1D index


n_res = 10
lat_release_NPSG = np.tile(np.linspace(23,32,n_res),[n_res,1])
lon_release_NPSG = np.tile(np.linspace(-143,-134,n_res),[n_res,1])
lons_NPSG, lats_NPSG = np.meshgrid(lon_release_NPSG, lat_release_NPSG)

lat_release_EqPac = np.tile(np.linspace(-4.5,4.5,n_res),[n_res,1])
lon_release_EqPac = np.tile(np.linspace(-148,-139,n_res),[n_res,1])
lons_EqPac, lats_EqPac = np.meshgrid(lon_release_EqPac, lat_release_EqPac)

lat_release_SO  = np.tile(np.linspace(-62,-53,n_res),[n_res,1])
lon_release_SO = np.tile(np.linspace(-115,-106,n_res),[n_res,1])
lons_SO, lats_SO = np.meshgrid(lon_release_SO, lat_release_SO)

lons = {'NPSG': lon_release_NPSG,
        'EqPac': lon_release_EqPac,
        'SO': lon_release_SO}
lats = {'NPSG': lat_release_NPSG,
        'EqPac': lat_release_EqPac,
        'SO': lat_release_SO}

iy_min, ix_min = getclosest_ij(mesh_mask['nav_lat'], mesh_mask['nav_lon'], lats[region][0,0], lons[region][0,0])
iy_max, ix_max = getclosest_ij(mesh_mask['nav_lat'], mesh_mask['nav_lon'], lats[region][0,-1], lons[region][0,-1])

D_region = ds_p_NEMO['PHD'].isel(y=slice(iy_min,iy_max),x=slice(ix_min,ix_max))
PP_region = ds_pp_NEMO['TPP3'].isel(y=slice(iy_min,iy_max),x=slice(ix_min,ix_max))
T_region = ds_ts_NEMO['potemp'].isel(y=slice(iy_min,iy_max),x=slice(ix_min,ix_max))
SAL_region = ds_ts_NEMO['salin'].isel(y=slice(iy_min,iy_max),x=slice(ix_min,ix_max))
TAU_region = ds_ts_NEMO['taum'].isel(y=slice(iy_min,iy_max),x=slice(ix_min,ix_max))
MLD_region = ds_ts_NEMO['mldr10_1'].isel(y=slice(iy_min,iy_max),x=slice(ix_min,ix_max))
W10_region = ds_ts_NEMO['sowindsp'].isel(y=slice(iy_min,iy_max),x=slice(ix_min,ix_max))

for i, filename in enumerate(ppfiles_NEMO[1:]):
    ds_0 = xr.open_dataset(filename)
    PP_0 = ds_0['TPP3'].isel(y=slice(iy_min,iy_max),x=slice(ix_min,ix_max))
    print(i)
    PP_region = xr.concat([PP_region,PP_0], 'time_counter')

for i, filename in enumerate(pfiles_NEMO[1:]):
    ds_0 = xr.open_dataset(filename)
    D_0 = ds_0['PHD'].isel(y=slice(iy_min,iy_max),x=slice(ix_min,ix_max))
    print(i)
    D_region = xr.concat([D_region,D_0], 'time_counter')

for i, filename in enumerate(tsfiles_NEMO[1:]):
    ds_0 = xr.open_dataset(filename)
    MLD_0 = ds_0['mldr10_1'].isel(y=slice(iy_min,iy_max),x=slice(ix_min,ix_max))
    print(i)
    MLD_region = xr.concat([MLD_region,MLD_0], 'time_counter')

for i, filename in enumerate(tsfiles_NEMO[1:]):
    ds_0 = xr.open_dataset(filename)
    TAU_0 = ds_0['taum'].isel(y=slice(iy_min,iy_max),x=slice(ix_min,ix_max))
    print(i)
    TAU_region = xr.concat([TAU_region,TAU_0], 'time_counter')

for i, filename in enumerate(tsfiles_NEMO[1:]):
    ds_0 = xr.open_dataset(filename)
    W10_0 = ds_0['sowindsp'].isel(y=slice(iy_min,iy_max),x=slice(ix_min,ix_max))
    print(i)
    W10_region = xr.concat([W10_region,W10_0], 'time_counter')

for i, filename in enumerate(tsfiles_NEMO[1:]):
    ds_0 = xr.open_dataset(filename)
    T_0 = ds_0['potemp'].isel(y=slice(iy_min,iy_max),x=slice(ix_min,ix_max))
    print(i)
    T_region = xr.concat([T_region,T_0], 'time_counter')

for i, filename in enumerate(tsfiles_NEMO[1:]):
    ds_0 = xr.open_dataset(filename)
    SAL_0 = ds_0['salin'].isel(y=slice(iy_min,iy_max),x=slice(ix_min,ix_max))
    print(i)
    SAL_region = xr.concat([SAL_region,SAL_0], 'time_counter')

PP_profile = PP_region.mean('time_counter').mean('y').mean('x')
PP_profile = np.nan_to_num(PP_profile)

D_profile = D_region.mean('time_counter').mean('y').mean('x')
D_profile = np.nan_to_num(D_profile)

Z = - ds_ts_NEMO['deptht']
SA = SAL_region
CT = T_region

SAu = 40 * 35.16504 / 35
CTu = 40
Zu = 1e4
deltaS = 32
R000 = 8.0189615746e+02
R100 = 8.6672408165e+02
R200 = -1.7864682637e+03
R300 = 2.0375295546e+03
R400 = -1.2849161071e+03
R500 = 4.3227585684e+02
R600 = -6.0579916612e+01
R010 = 2.6010145068e+01
R110 = -6.5281885265e+01
R210 = 8.1770425108e+01
R310 = -5.6888046321e+01
R410 = 1.7681814114e+01
R510 = -1.9193502195e+00
R020 = -3.7074170417e+01
R120 = 6.1548258127e+01
R220 = -6.0362551501e+01
R320 = 2.9130021253e+01
R420 = -5.4723692739e+00
R030 = 2.1661789529e+01
R130 = -3.3449108469e+01
R230 = 1.9717078466e+01
R330 = -3.1742946532e+00
R040 = -8.3627885467e+00
R140 = 1.1311538584e+01
R240 = -5.3563304045e+00
R050 = 5.4048723791e-01
R150 = 4.8169980163e-01
R060 = -1.9083568888e-01
R001 = 1.9681925209e+01
R101 = -4.2549998214e+01
R201 = 5.0774768218e+01
R301 = -3.0938076334e+01
R401 = 6.6051753097e+00
R011 = -1.3336301113e+01
R111 = -4.4870114575e+00
R211 = 5.0042598061e+00
R311 = -6.5399043664e-01
R021 = 6.7080479603e+00
R121 = 3.5063081279e+00
R221 = -1.8795372996e+00
R031 = -2.4649669534e+00
R131 = -5.5077101279e-01
R041 = 5.5927935970e-01
R002 = 2.0660924175e+00
R102 = -4.9527603989e+00
R202 = 2.5019633244e+00
R012 = 2.0564311499e+00
R112 = -2.1311365518e-01
R022 = -1.2419983026e+00
R003 = -2.3342758797e-02
R103 = -1.8507636718e-02
R013 = 3.7969820455e-01
ss = np.sqrt((SA + deltaS) / SAu)
tt = CT / CTu
zz = -Z / Zu
rz3 = R013 * tt + R103 * ss + R003
rz2 = (R022 * tt + R112 * ss + R012) * tt + (R202 * ss + R102) * ss + R002
rz1 = (((R041 * tt + R131 * ss + R031) * tt + (R221 * ss + R121) * ss + R021) * tt + ((R311 * ss + R211) * ss + R111) * ss + R011) * tt + (((R401 * ss + R301) * ss + R201) * ss + R101) * ss + R001
rz0 = (((((R060 * tt + R150 * ss + R050) * tt + (R240 * ss + R140) * ss + R040) * tt + ((R330 * ss + R230) * ss + R130) * ss + R030) * tt + (((R420 * ss + R320) * ss + R220) * ss + R120) * ss + R020) * tt + ((((R510 * ss + R410) * ss + R310) * ss + R210) * ss + R110) * ss + R010) * tt + (((((R600 * ss + R500) * ss + R400) * ss + R300) * ss + R200) * ss + R100) * ss + R000
RHO_region = ((rz3 * zz + rz2) * zz + rz1) * zz + rz0

g = 7.32e10/(86400.**2.)
rho_a = 1.22
wave_age = 35
phi = 0.9
vk = 0.4

u_s_a =  np.sqrt(TAU_region/rho_a)
u_s_w =  np.sqrt(np.divide(TAU_region,RHO_region.isel(deptht=0)))

alpha = (vk * u_s_w) / phi

beta = np.divide((wave_age * u_s_a), W10_region)
z0 = 3.5153e-5 * np.power(beta, -0.42) * np.square(W10_region) / g

KPP_region = alpha * (ds_ts_NEMO['deptht'] + z0) * np.square(1-np.divide(ds_ts_NEMO['deptht'],MLD_region))

Z_region = np.tile(ds_ts_NEMO['deptht'],(KPP_region.shape[2],1))
Z_region = np.tile(Z_region, (KPP_region.shape[1],1,1))
Z_region = np.tile(Z_region, (KPP_region.shape[0],1,1,1))

mld_region = np.tile(np.expand_dims(MLD_region,axis=3), (1,1,1,len(ds_ts_NEMO['deptht'])))

kpp_region = KPP_region.values
kpp_region[Z_region>mld_region] = 0

RHO_profile = RHO_region.mean('time_counter').mean('y').mean('x')
RHO_profile = np.nan_to_num(RHO_profile)

kpp_profile = np.mean(kpp_region, axis=(0,1,2))

climatology = np.array([D_profile, PP_profile, kpp_profile])
np.save('/data/oceanparcels/output_data/data_Reint/'+region+'_climatology', climatology)
