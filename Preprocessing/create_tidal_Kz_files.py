"""
Author: Reint Fischer

Executable python file with functions that compute tidal Kz netCDF4 files from global climatology of tidal turbulent kinetic energy.
"""

import numpy as np
from numpy import array
import scipy.interpolate as interpolate
import xarray
from netCDF4 import Dataset

def create_tidal_Kz_files(file_name: str):
    """
    Author: Victor Onink
    adapted for biofouling_3dtransport_2 by Reint Fischer

    Determining Kz and its vertical derivative from the estimates of production of turbulent kinetic energy, as outlined in
    de Lavergne et al. (2020) and based on code shared by Clement Vic. The input file for TIDAL_filename can be
    downloaded at https://www.seanoe.org/data/00619/73082/
    """
    # Loading the global data from de Lavergne et al. (2020)
    datadir = '/scratch/dlobelle/Kooi_data/data_input/'
    TIDAL_filename = datadir + 'tidal_mixing_3D_maps.nc'
    TIDAL_data = {}
    for key in Dataset(TIDAL_filename).variables.keys():
        TIDAL_data[key] = Dataset(TIDAL_filename).variables[key][:]

    # Computing Kz on the TIDAL_data grid according to Kv = gamma * epsilon / N^2
    gamma = 0.2  # Mixing efficiency
    TIDAL_Kz = np.divide(gamma * TIDAL_data['epsilon_tid'], TIDAL_data['buoyancy_frequency_squared'])

    # The TIDAL_data gridding isn't regular in the z-direction. The seafloor divides cells into partial cells.
    # Since we are not interested in the exact position of the seafloor, and Parcels cannot easily handle 3D depth coordinates,
    # we reposition all cells onto the regular depth intervals.
    
    # Create 1D depth profile
    depth_id = np.argwhere(TIDAL_data['depth_midpoint']==np.nanmax(TIDAL_data['depth_midpoint']))[0,1:] 
    depths = TIDAL_data['depth_midpoint'][:,depth_id[0],depth_id[1]].data
    # Extrapolate surface to depth = 0m.
    TIDAL_Kz_ext = np.insert(TIDAL_Kz, 0, TIDAL_Kz[0], axis=0)
    depths = np.insert(depths, 0, [0])
    
    # rewrite longitude to -180 - 180
    lons = TIDAL_data['lon']
    lons[lons>180] = lons[lons>180]-360
    
    # Sort longitudes and corresponding data
    ids = np.argsort(lons)
    lons = lons[ids]
    TIDAL_Kz_ext = TIDAL_Kz_ext[:,:,ids]
    
    # Extrapolate Kz to longitude = -180 to prevent sampling out of bounds.
    TIDAL_Kz_ext = np.insert(TIDAL_Kz_ext, 0, TIDAL_Kz_ext[:,:,0], axis=2)
    lons = np.insert(lons, 0, [-180.])

    # Due to very low N^2 values (corresponding to very weak stratification), there are some regions where Kz is
    # unfeasibly high (Kz > 100 m^2/s). Therefore, I cap GRID_Kz at 0.1 m^2/s. This only affects 0.08% of all the cells
    # in the TIDAL_data, and prevents numerical issues later on.
    TIDAL_Kz_ext[TIDAL_Kz_ext > 1e-1] = 1e-1

    # Calculate the vertical gradient of Kz
    TIDAL_dKz_ext = np.gradient(TIDAL_Kz_ext,depths, axis=0)

    # Saving the field to a .nc file
    coords = {'Latitude': (['Latitude'], TIDAL_data['lat']),
              'Longitude': (['Longitude'], lons),
              'Depth_midpoint':(['Depth_midpoint'], depths)}
    dims = ('Depth_midpoint', 'Latitude', 'Longitude')
    dset = xarray.Dataset({'TIDAL_Kz': xarray.DataArray(TIDAL_Kz_ext, dims=dims, coords=coords),'TIDAL_dKz':xarray.DataArray(TIDAL_dKz_ext, dims=dims, coords=coords)}, coords=coords)
    dset.to_netcdf(file_name)

if __name__ == "__main__":
    create_tidal_Kz_files('/scratch/dlobelle/Kooi_data/data_input/Kz.nc')
