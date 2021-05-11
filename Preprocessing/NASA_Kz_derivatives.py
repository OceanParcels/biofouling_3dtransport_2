"""
Author: Reint Fischer

Executable python file with functions that compute the vertical derivative of Kz from the NASA-GISS modelE and store it in a netCDF
"""

import numpy as np
from numpy import array
from glob import glob
import scipy.interpolate as interpolate
import xarray
from netCDF4 import Dataset

def create_dKz_from_NASA(filedir: str):
    """
    Author: Reint Fischer

    Determining dKzdz
    """
    # Loading the global data from de Lavergne et al. (2020)
    datadir = '/data/oceanparcels/input_data/NASA_GISS/modelE/'
    filelist = (sorted(glob(datadir + '*oijlh240Earobio3_hiAR6subdd.nc')))
    
    for i, filename in enumerate(filelist):
        dataset = xarray.open_dataset(filename)
        
        Kz = dataset['kvert']
        depths = dataset['zoc']

        # Calculate the vertical gradient of Kz
        dKz = np.gradient(Kz,depths, axis=1)

        # Saving the field to a .nc file
        coords = {'latitude': (['latitude'], dataset['lato']),
                  'longitude': (['longitude'], dataset['lono']),
                  'depth':(['depth'], depths),
                  'time': (['time'], dataset['time'])}
        dims = ('time','depth', 'latitude', 'longitude')
        dset = xarray.Dataset({'dKz': xarray.DataArray(dKz, dims=dims, coords=coords)}, coords=coords)
        dset.to_netcdf(filedir+filename[len(datadir):len(datadir)+8]+'_dKz.nc')
        dset.close()
        print(filename)

print('python file')
create_dKz_from_NASA('/data/oceanparcels/input_data/NASA_GISS/dKz/')
