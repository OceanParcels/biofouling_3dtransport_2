from parcels import FieldSet, ParticleSet, JITParticle, ScipyParticle, AdvectionRK4_3D, AdvectionRK4, ErrorCode, ParticleFile, Variable, Field, NestedField, VectorField, timer, ParcelsRandom 
from parcels.kernels.TEOSseawaterdensity import PolyTEOS10_bsq
from datetime import timedelta as delta
from datetime import  datetime
import numpy as np
from numpy.random import default_rng
import math
from glob import glob
import os
import xarray as xr
import sys
import time as timelib
import matplotlib.pyplot as plt
import warnings
import pickle                                                      
import matplotlib.ticker as mtick
import pandas as pd 
import operator
from numpy import *
import scipy.linalg
import math as math
from argparse import ArgumentParser
warnings.filterwarnings("ignore")

seed = 123
ParcelsRandom.seed(seed)
rng = default_rng(seed)

#------ Choose ------:
simdays = 20 #90
secsdt = 60 #30
hrsoutdt = 12 #2

"""functions and kernels"""

def Kooi(particle,fieldset,time):  
    """
    Kernel to compute the vertical velocity (Vs) of particles due to changes in ambient algal concentrations, growth and death of attached algae based on Kooi et al. 2017 model 
    """
    
    #------ Nitrogen to cell ratios for ambient algal concentrations ('aa') and algal growth ('mu_aa') from NEMO output (no longer using N:C:AA (Redfield ratio), directly N:AA from Menden-Deuer and Lessard 2000)     
    min_N2cell = 2656.0e-09 #[mgN cell-1] (from Menden-Deuer and Lessard 2000)
    max_N2cell = 11.0e-09   #[mgN cell-1] 
    med_N2cell = 356.04e-09 #[mgN cell-1] median value is used below (as done in Kooi et al. 2017)
      
    #------ Ambient algal concentration from MEDUSA's diatom phytoplankton 
    n0 = particle.d_phy # [mmol N m-3] in MEDUSA
    n = n0*14.007                       # conversion from [mmol N m-3] to [mg N m-3] (atomic weight of 1 mol of N = 14.007 g)   
    n2 = n/med_N2cell                   # conversion from [mg N m-3] to [no. m-3]
    
    if n2<0.: 
        aa = 0.
    else:
        aa = n2                        # [no m-3] to compare to Kooi model    
    
    #------ Primary productivity (algal growth) from MEDUSA TPP3 (no longer condition of only above euphotic zone, since not much diff in results)
    tpp0 = particle.tpp3              # [mmol N m-3 d-1]
    mu_n0 = tpp0*14.007               # conversion from [mmol N m-3 d-1] to [mg N m-3 d-1] (atomic weight of 1 mol of N = 14.007 g) 
    mu_n = mu_n0/med_N2cell           # conversion from [mg N m-3 d-1] to [no. m-3 d-1]
    mu_n2 = mu_n/aa                   # conversion from [no. m-3 d-1] to [d-1]
    
    if mu_n2<0.:
        mu_aa = 0.
    else:
        mu_aa = mu_n2/86400.          # conversion from d-1 to s-1
    
    #------ Profiles from MEDUSA or Kooi theoretical profiles -----
    z = particle.depth           # [m]
    t = particle.temp            # [oC]
    sw_visc = particle.sw_visc   # [kg m-1 s-1]
    kin_visc = particle.kin_visc # [m2 s-1]
    rho_sw = particle.density    # [kg m-3]       
    a = particle.a               # [no. m-2 s-1]
    vs = particle.vs             # [m s-1]   

    #------ Constants and algal properties -----
    g = 7.32e10/(86400.**2.)    # gravitational acceleration (m d-2), now [s-2]
    k = 1.0306E-13/(86400.**2.) # Boltzmann constant [m2 kg d-2 K-1] now [s-2] (=1.3804E-23)
    rho_bf = 1388.              # density of biofilm ([g m-3]
    v_a = 2.0E-16               # Volume of 1 algal cell [m-3]
    m_a = fieldset.mortality_rate/86400. # mortality rate, now [s-1]
    r20 = 0.1/86400.            # respiration rate, now [s-1] 
    q10 = 2.                    # temperature coefficient respiration [-]
    gamma = 1.728E5/86400.      # shear [d-1], now [s-1]
    
    #------ Volumes -----
    v_pl = (4./3.)*math.pi*particle.r_pl**3.             # volume of plastic [m3]
    theta_pl = 4.*math.pi*particle.r_pl**2.              # surface area of plastic particle [m2]
    r_a = ((3./4.)*(v_a/math.pi))**(1./3.)      # radius of algae [m]
    
    v_bf = (v_a*a)*theta_pl                           # volume of biofilm [m3]
    v_tot = v_bf + v_pl                               # volume of total [m3]
    t_bf = ((v_tot*(3./(4.*math.pi)))**(1./3.))-particle.r_pl  # biofilm thickness [m] 
    
    #------ Diffusivity -----
    r_tot = particle.r_pl + t_bf                               # total radius [m]
    rho_tot = (particle.r_pl**3. * particle.rho_pl + ((particle.r_pl + t_bf)**3. - particle.r_pl**3.)*rho_bf)/(particle.r_pl + t_bf)**3. # total density [kg m-3]
    theta_tot = 4.*math.pi*r_tot**2.                          # surface area of total [m2]
    d_pl = k * (t + 273.16)/(6. * math.pi * sw_visc * r_tot)  # diffusivity of plastic particle [m2 s-1]
    d_a = k * (t + 273.16)/(6. * math.pi * sw_visc * r_a)     # diffusivity of algal cells [m2 s-1] 
    
    #------ Encounter rates -----
    beta_abrown = 4.*math.pi*(d_pl + d_a)*(r_tot + r_a)       # Brownian motion [m3 s-1] 
    beta_ashear = 1.3*gamma*((r_tot + r_a)**3.)               # advective shear [m3 s-1]
    beta_aset = (1./2.)*math.pi*r_tot**2. * abs(vs)           # differential settling [m3 s-1]
    beta_a = beta_abrown + beta_ashear + beta_aset            # collision rate [m3 s-1]
    
    #------ Attached algal growth (Eq. 11 in Kooi et al. 2017) -----
    a_coll = (beta_a*aa)/theta_pl*fieldset.collision_eff
    a_growth = mu_aa*a
    a_mort = m_a*a
    a_resp = (q10**((t-20.)/10.))*r20*a     
    
    particle.a += (a_coll + a_growth - a_mort - a_resp) * particle.dt

    dn = 2. * (r_tot)                             # equivalent spherical diameter [m]
    delta_rho = (rho_tot - rho_sw)/rho_sw         # normalised difference in density between total plastic+bf and seawater[-]        
    dstar = ((rho_tot - rho_sw) * g * dn**3.)/(rho_sw * kin_visc**2.) # [-]
    
        
    if dstar > 5e9:
        w = 1000.
    elif dstar <0.05:
        w = (dstar**2.) *1.71E-4
    else:
        w = 10.**(-3.76715 + (1.92944*math.log10(dstar)) - (0.09815*math.log10(dstar)**2.) - (0.00575*math.log10(dstar)**3.) + (0.00056*math.log10(dstar)**4.))
    
    #------ Settling of particle -----
#     if z >= 4000.: 
#         vs = 0 
#     elif z<=0.6 and delta_rho < 0:# for rising particles above 0.6 m (negative delta_rho = rho_tot < rho_sw)
#         vs = 0
#         particle.depth = 0.6
    if delta_rho > 0: # sinks 
        vs = (g * kin_visc * w * delta_rho)**(1./3.)
    else: #rises 
        a_del_rho = delta_rho*-1.
        vs = -1.*(g * kin_visc * w * a_del_rho)**(1./3.)  # m s-1
    
    particle.vs_init = vs
    
    z0 = z + vs * particle.dt 
    if z0 <=0.6 or z0 >= 4000.: # NEMO's 'surface depth'
        vs = 0
        particle.depth = 0.6
    else:          
        particle.depth += vs * particle.dt 

    particle.vs = vs
    particle.rho_tot = rho_tot
    particle.r_tot = r_tot
#     particle.t_bf = t_bf
#     particle.beta_a = beta_a
#     particle.beta_abrown = beta_abrown
#     particle.beta_ashear = beta_ashear
#     particle.beta_aset = beta_aset
    particle.delta_rho = delta_rho

def Kooi_no_biofouling(particle,fieldset,time):  
    """
    Kernel to compute the vertical velocity (Vs) of particles due to their different sizes and densities (removed all effects of changes in ambient algal concentrations, growth and death of attached algae)- based on Kooi et al. 2017 model 
    """
    
    #------ Profiles from MEDUSA or Kooi theoretical profiles -----
    z = particle.depth           # [m]
    t = particle.temp            # [oC]
    kin_visc = particle.kin_visc # kinematic viscosity[m2 s-1]
    rho_sw = particle.density    # seawater density[kg m-3]       
    vs = particle.vs             # vertical velocity[m s-1]   

    #------ Constants -----
    g = 7.32e10/(86400.**2.)    # gravitational acceleration (m d-2), now [s-2]
    
    #------ Volumes -----
    v_pl = (4./3.)*math.pi*particle.r_pl**3.             # volume of plastic [m3]
    theta_pl = 4.*math.pi*particle.r_pl**2.              # surface area of plastic particle [m2]    

    #------ Diffusivity -----
    r_tot = particle.r_pl #+ t_bf                               # total radius [m]
    rho_tot = (particle.r_pl**3. * particle.rho_pl)/(particle.r_pl)**3. # total density [kg m-3]

    dn = 2. * (r_tot)                             # equivalent spherical diameter [m]
    delta_rho = (rho_tot - rho_sw)/rho_sw         # normalised difference in density between total plastic+bf and seawater[-]        
    dstar = ((rho_tot - rho_sw) * g * dn**3.)/(rho_sw * kin_visc**2.) # dimensional diameter[-]
            
    if dstar > 5e9:
        w = 1000.
    elif dstar <0.05:
        w = (dstar**2.) *1.71E-4
    else:
        w = 10.**(-3.76715 + (1.92944*math.log10(dstar)) - (0.09815*math.log10(dstar)**2.) - (0.00575*math.log10(dstar)**3.) + (0.00056*math.log10(dstar)**4.))
        
    #------ Settling of particle -----

    if delta_rho > 0: # sinks 
        vs = (g * kin_visc * w * delta_rho)**(1./3.)
    else: #rises 
        a_del_rho = delta_rho*-1.
        vs = -1.*(g * kin_visc * w * a_del_rho)**(1./3.)  # m s-1
    
    particle.vs_init = vs
    z0 = z + vs * particle.dt 
    if z0 <=0.6 or z0 >= 4000.: # NEMO's 'surface depth'
        vs = 0
        particle.depth = 0.6
    else:          
        particle.depth += vs * particle.dt 

    particle.vs = vs
    particle.rho_tot = rho_tot
    particle.r_tot = r_tot
    particle.delta_rho = delta_rho
    
def DeleteParticle(particle, fieldset, time):
    """Kernel for deleting particles if they are out of bounds."""
    print('particle is deleted at lon = '+str(particle.lon)+', lat ='+str(particle.lat)+', depth ='+str(particle.depth)) #print(particle.lon, particle.lat, particle.depth)
    particle.delete() 
    
def getclosest_ij(lats,lons,latpt,lonpt):     
    """Function to find the index of the closest point to a certain lon/lat value."""
    dist_sq = (lats-latpt)**2 + (lons-lonpt)**2                 # find squared distance of every point on grid
    minindex_flattened = dist_sq.argmin()                       # 1D index of minimum dist_sq element
    return np.unravel_index(minindex_flattened, lats.shape)     # Get 2D index for latvals and lonvals arrays from 1D index

def periodicBC(particle, fieldset, time):
    if particle.lon < 0.:
        particle.lon += 360.
    elif particle.lon >= 360.:
        particle.lon -= 360.
           
def Profiles(particle, fieldset, time):  
    particle.temp = fieldset.cons_temperature[time, particle.depth,particle.lat,particle.lon]  
    particle.d_phy= fieldset.d_phy[time, particle.depth,particle.lat,particle.lon] 
    particle.tpp3 = fieldset.tpp3[time,particle.depth,particle.lat,particle.lon]
    
    mu_w = 4.2844E-5 + (1/((0.157*(particle.temp + 64.993)**2)-91.296))
    A = 1.541 + 1.998E-2*particle.temp - 9.52E-5*particle.temp**2
    B = 7.974 - 7.561E-2*particle.temp + 4.724E-4*particle.temp**2
    S_sw = fieldset.abs_salinity[time, particle.depth, particle.lat, particle.lon]
    particle.sw_visc = mu_w*(1 + A*S_sw + B*S_sw**2)
    particle.kin_visc = particle.sw_visc/particle.density
    particle.w = fieldset.W[time,particle.depth,particle.lat,particle.lon]
    
def select_from_Cozar_random_continuous(number_of_particles, e_max=-3, e_min=-6):
    '''
    Create a set of particle radii by randomly drawing from a continuous distribution.
    UNFINISHED
    '''
    r_pls = rng.power(3, number_of_particles)
    return r_pls

def select_from_Cozar_determined(number_of_particles, e_max=-3, e_min=-6):
    '''
    Create a set of particle radii according to the Cozar distribution.
    :param number_of_particles: Size of particleset
    :param e_max: Exponent of the largest particle. -3 -> 1E-3 m = 1 mm
    :param e_min: Exponent of the smallest particle. -6 -> 1E-6 = 1 um
    '''
    nbins = e_max-e_min+1
    bins = np.logspace(e_min, e_max, nbins)
    distribution = bins[-1]**2/(bins**2)
    particles_per_bin = distribution/np.sum(distribution)
    particles_per_bin = particles_per_bin.round().astype(int)
    r_pls = []
    for i,r in enumerate(bins):
        r_pls += [r]*particles_per_bin[i]
    return r_pls

def vertical_mixing_random_constant(particle, fieldset, time):
    if particle.depth < fieldset.mldr[time, particle.depth, particle.lat, particle.lon]:
        particle.mld = 1                           # Particle is in mixed layer
        vmax = 0.2                                 # [m/s] Maximum velocity
        particle.w_m = vmax*2*(ParcelsRandom.random()-0.5)  # [m/s] vertical mixing velocity
        z_0 = particle.depth + particle.w_m*particle.dt
        if z_0 <= 0.6:                              # [m] NEMO's surface depth
            particle.depth = 0.6
        else:
            particle.depth = z_0
    else:
        particle.mld = 0
        particle.w_m = 0

def mixed_layer(particle, fieldset, time):
    if particle.depth < fieldset.mldr[time, particle.depth, particle.lat, particle.lon]:
        particle.mld = 1
    else:
        particle.mld = 0

def markov_0_reflect(particle, fieldset, time):
    """
    If a particle tries to cross the boundary, then it is reflected back
    Author: Victor Onink
    Adapted 1D -> 3D
    """
    # According to Ross & Sharples (2004), first the deterministic part of equation 1
    
""" Defining the particle class """

class plastic_particle(JITParticle): #ScipyParticle): #
    u = Variable('u', dtype=np.float32,to_write=True)
    v = Variable('v', dtype=np.float32,to_write=True)
    w = Variable('w', dtype=np.float32,to_write=True)
    temp = Variable('temp',dtype=np.float32,to_write=False)
    density = Variable('density',dtype=np.float32,to_write=True)
    tpp3 = Variable('tpp3',dtype=np.float32,to_write=False)
    d_phy = Variable('d_phy',dtype=np.float32,to_write=False)
    a = Variable('a',dtype=np.float32,to_write=True)
    kin_visc = Variable('kin_visc',dtype=np.float32,to_write=False)
    sw_visc = Variable('sw_visc',dtype=np.float32,to_write=False)
    vs = Variable('vs',dtype=np.float32,to_write=True)
    w_m = Variable('w_m', dtype=np.float32, to_write=True)
    mld = Variable('mld', dtype=np.float32, to_write=True) 
    rho_tot = Variable('rho_tot',dtype=np.float32,to_write=True) 
    r_tot = Variable('r_tot',dtype=np.float32,to_write=True)
    delta_rho = Variable('delta_rho',dtype=np.float32,to_write=True)
    vs_init = Variable('vs_init',dtype=np.float32,to_write=True)
    r_pl = Variable('r_pl',dtype=np.float32,to_write='once')   
    rho_pl = Variable('rho_pl',dtype=np.float32,to_write='once')
    
if __name__ == "__main__":     
    p = ArgumentParser(description="""choose starting month and year""")
    p.add_argument('-mon', choices = ('01','12','03','06','09'), action="store", dest="mon", 
                   help='start month for the run')
    p.add_argument('-yr', choices = ('2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010'), action="store", dest="yr",
                   help='start year for the run')
    p.add_argument('-region', choices = ('GPGP','EqPac'), action = "store", dest = "region",
                   help ='region where particles released')
    p.add_argument('-a_mort', choices = ('0.16', '0.39', '0.5'), action = "store", dest = 'mortality_rate', help='Mortality rate in d-1')
    p.add_argument('-mixing', choices = ('no', 'fixed'), action = "store", dest = 'mixing', help='Type of random vertical mixing. "no" is none, "fixed" is mld between 0.2 and -0.2 m/s')
    p.add_argument('-collision_eff', choices = ('1', '0.5'), default='1', action='store', dest='collision_eff', help='Collision efficiency: fraction of colliding algae that stick to the particle')
    p.add_argument('-system', choices=('gemini', 'cartesius'), action='store', dest = 'system', help='"gemini" or "cartesius"')
    

    args = p.parse_args()
    mon = args.mon
    yr = args.yr
    region = args.region
    mortality_rate = float(args.mortality_rate)
    mixing = args.mixing
    collision_eff = float(args.collision_eff)
    no_biofouling = False #no_biofouling = args.no_biofouling
    no_advection = False #no_advection = args.no_advection
    system = args.system
    
    """ Load particle release locations from plot_NEMO_landmask.ipynb """
    # CHOOSE

    #------ Fieldset grid  ------
    if region == 'GPGP':
        minlat = 20 
        maxlat = 45 
        minlon = 110 # -180 #75 
        maxlon = -120 #45   
    elif region == 'EqPac':
        minlat = -20 
        maxlat = 20
        minlon = -180
        maxlon = -120


    #------ Release particles on a 10x10 deg grid ------
    if region == 'GPGP':
        lat_release0 = np.tile(np.linspace(28,36,50),[50,1]) #(20,28,5),[5,1]) 
        lat_release = lat_release0.T 
        lon_release = np.tile(np.linspace(-135,-143,50),[50,1]) #(-140,-148,5),[5,1]) 
    elif region == 'EqPac':
        lat_release0 = np.tile(np.linspace(-4,4,50),[50,1]) 
        lon_release = np.tile(np.linspace(-140,-148,50),[50,1])
        lat_release = lat_release0.T 
    z_release = np.tile(0.6,[50,50]) 
    res = '0.2x0.2' 
    
    """ Defining the fieldset""" 
    if system == 'cartesius':
        dirread = '/projects/0/topios/hydrodynamic_data/NEMO-MEDUSA/ORCA0083-N006/means/'
        dirread_bgc = '/projects/0/topios/hydrodynamic_data/NEMO-MEDUSA_BGC/ORCA0083-N006/means/'  
        dirread_mesh = '/projects/0/topios/hydrodynamic_data/NEMO-MEDUSA/ORCA0083-N006/domain/'
    elif system == 'gemini':
        dirread = '/data/oceanparcels/input_data/NEMO-MEDUSA/ORCA0083-N006/means/'
        dirread_bgc = '/data/oceanparcels/input_data/NEMO-MEDUSA_BGC/ORCA0083-N006/means/'
        dirread_mesh = '/data/oceanparcels/input_data/NEMO-MEDUSA/ORCA0083-N006/domain/'
    else:
        print('Error: no valid system argument parsed')  

    if mon =='12':
        yr0 = str(int(yr)-1)
        ufiles = (sorted(glob(dirread+'ORCA0083-N06_'+yr0+'1*d05U.nc'))+ sorted(glob(dirread+'ORCA0083-N06_'+yr+'*d05U.nc')))
        vfiles = (sorted(glob(dirread+'ORCA0083-N06_'+yr0+'1*d05V.nc'))+ sorted(glob(dirread+'ORCA0083-N06_'+yr+'*d05V.nc')))
        wfiles = (sorted(glob(dirread+'ORCA0083-N06_'+yr0+'1*d05W.nc'))+ sorted(glob(dirread+'ORCA0083-N06_'+yr+'*d05W.nc')))
        pfiles = (sorted(glob(dirread_bgc+'ORCA0083-N06_'+yr0+'1*d05P.nc'))+ sorted(glob(dirread_bgc+'ORCA0083-N06_'+yr+'*d05P.nc')))
        ppfiles = (sorted(glob(dirread_bgc+'ORCA0083-N06_'+yr0+'1*d05D.nc'))+ sorted(glob(dirread_bgc+'ORCA0083-N06_'+yr+'*d05D.nc')))
        tsfiles = (sorted(glob(dirread+'ORCA0083-N06_'+yr0+'1*d05T.nc'))+ sorted(glob(dirread+'ORCA0083-N06_'+yr+'*d05T.nc')))
    elif mon == '01':
        yr0 = yr
        ufiles = sorted(glob(dirread+'ORCA0083-N06_'+yr+mon+'*d05U.nc'))
        vfiles = sorted(glob(dirread+'ORCA0083-N06_'+yr+mon+'*d05V.nc'))
        wfiles = sorted(glob(dirread+'ORCA0083-N06_'+yr+mon+'*d05W.nc'))
        pfiles = sorted(glob(dirread_bgc+'ORCA0083-N06_'+yr+mon+'*d05P.nc'))
        ppfiles = sorted(glob(dirread_bgc+'ORCA0083-N06_'+yr+mon+'*d05D.nc'))
        tsfiles = sorted(glob(dirread+'ORCA0083-N06_'+yr+mon+'*d05T.nc'))
    else:
        yr0 = yr
        ufiles = sorted(glob(dirread+'ORCA0083-N06_'+yr+'*d05U.nc')) 
        vfiles = sorted(glob(dirread+'ORCA0083-N06_'+yr+'*d05V.nc')) 
        wfiles = sorted(glob(dirread+'ORCA0083-N06_'+yr+'*d05W.nc')) 
        pfiles = sorted(glob(dirread_bgc+'ORCA0083-N06_'+yr+'*d05P.nc')) 
        ppfiles = sorted(glob(dirread_bgc+'ORCA0083-N06_'+yr+'*d05D.nc')) 
        tsfiles = sorted(glob(dirread+'ORCA0083-N06_'+yr+'*d05T.nc')) 
        
    mesh_mask = dirread_mesh+'coordinates.nc'

    filenames = {'U': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': ufiles}, #'depth': wfiles,
                 'V': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': vfiles},
                 'W': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': wfiles},
                 'd_phy': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': pfiles},
                 'tpp3': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': ppfiles},
                 'cons_temperature': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': tsfiles},
                 'abs_salinity': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': tsfiles},
                 'mldr': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': tsfiles}}

    variables = {'U': 'uo',
                 'V': 'vo',
                 'W': 'wo',
                 'd_phy': 'PHD',
                 'tpp3': 'TPP3', # units: mmolN/m3/d 
                 'cons_temperature': 'potemp',
                 'abs_salinity': 'salin',
                 'mldr': 'mldr10_1'}

    dimensions = {'U': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'}, #time_centered
                  'V': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'},
                  'W': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'},
                  'd_phy': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw','time': 'time_counter'},
                  'tpp3': {'lon': 'glamf', 'lat': 'gphif','depth': 'depthw', 'time': 'time_counter'},
                  'cons_temperature': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw','time': 'time_counter'},
                  'abs_salinity': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw','time': 'time_counter'},
                  'mldr': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'}}
    
    initialgrid_mask = dirread+'ORCA0083-N06_20070105d05U.nc'
    mask = xr.open_dataset(initialgrid_mask, decode_times=False)
    Lat, Lon = mask.variables['nav_lat'], mask.variables['nav_lon']
    latvals = Lat[:]; lonvals = Lon[:] # extract lat/lon values to numpy arrays
                                                                                               
    iy_min, ix_min = getclosest_ij(latvals, lonvals, minlat-5, minlon)
    iy_max, ix_max = getclosest_ij(latvals, lonvals, maxlat+5, maxlon)

    indices = {'lat': range(iy_min, iy_max), 'lon': range(ix_min, ix_max)} #depth : range(0,2000)
    print(indices['lat']) 

    chs = {'U': {'time': ('time_counter', 1), 'depth': ('depthu', 25), 'lat': ('y', 200), 'lon': ('x', 200)},
           'V': {'time': ('time_counter', 1), 'depth': ('depthv', 25), 'lat': ('y', 200), 'lon': ('x', 200)},
           'W': {'time': ('time_counter', 1), 'depth': ('depthw', 25), 'lat': ('y', 200), 'lon': ('x', 200)},
           'd_phy': {'time': ('time_counter', 1), 'depth': ('deptht', 25), 'lat': ('y', 200), 'lon': ('x', 200)},
           'tpp3': {'time': ('time_counter', 1), 'depth': ('deptht', 25), 'lat': ('y', 200), 'lon': ('x', 200)},
           'cons_temperature': {'time': ('time_counter', 1), 'depth': ('deptht', 25), 'lat': ('y', 200), 'lon': ('x', 200)},
           'abs_salinity': {'time': ('time_counter', 1), 'depth': ('deptht', 25), 'lat': ('y', 200), 'lon': ('x', 200)},
           'mldr': {'time': ('time_counter', 1), 'depth': ('deptht', 25), 'lat': ('y', 200), 'lon': ('x', 200)}}
        
    fieldset = FieldSet.from_nemo(filenames, variables, dimensions, allow_time_extrapolation=False, chunksize=chs, indices = indices) 
    fieldset.add_constant('mortality_rate', mortality_rate)
    fieldset.add_constant('collision_eff', collision_eff)

    lons = fieldset.U.lon
    lats = fieldset.U.lat
    depths = fieldset.U.depth

    """ Defining the particle set """   
       
    rho_pls = [920, 920, 920, 920, 920]  # add/remove here if more needed
    r_pls = [0.5e-3, 1e-4, 1e-5, 1e-6]

    pset = ParticleSet.from_list(fieldset=fieldset,         # the fields on which the particles are advected
                                 pclass=plastic_particle,   # the type of particles (JITParticle or ScipyParticle)
                                 lon= lon_release, #-160.,  # a vector of release longitudes 
                                 lat= lat_release, #36., 
                                 time = np.datetime64('%s-%s-05' % (yr0, mon)),
                                 depth = z_release,
                                 r_pl = r_pls[0] * np.ones(np.array(lon_release).size),
                                 rho_pl = rho_pls[0] * np.ones(np.array(lon_release).size),
                                 r_tot = r_pls[0] * np.ones(np.array(lon_release).size),
                                 rho_tot = rho_pls[0] * np.ones(np.array(lon_release).size))

    for r_pl, rho_pl in zip(r_pls[1:], rho_pls[1:]):
        pset.add(ParticleSet.from_list(fieldset=fieldset,         # the fields on which the particles are advected
                                 pclass=plastic_particle,   # the type of particles (JITParticle or ScipyParticle)
                                 lon= lon_release, #-160.,  # a vector of release longitudes 
                                 lat= lat_release, #36., 
                                 time = np.datetime64('%s-%s-05' % (yr0, mon)),
                                 depth = z_release,
                                 r_pl = r_pl * np.ones(np.array(lon_release).size),
                                 rho_pl = rho_pl * np.ones(np.array(lon_release).size),
                                 r_tot = r_pl * np.ones(np.array(lon_release).size),
                                 rho_tot = rho_pl * np.ones(np.array(lon_release).size)))


    """ Kernal + Execution"""
    if mon=='12':
        s = 'DJF'
    elif mon=='03':
        s = 'MAM'
    elif mon=='06':
        s = 'JJA'
    elif mon=='09':
        s = 'SON'
    elif mon=='01':
        s = 'Jan'
    
    kernels = pset.Kernel(AdvectionRK4_3D) 
    if mixing == 'fixed':
        kernels += pset.Kernel(vertical_mixing_random_constant)
    else:
        kernels += pset.Kernel(mixed_layer)
    kernels += pset.Kernel(PolyTEOS10_bsq) + pset.Kernel(Profiles) + pset.Kernel(Kooi) 
    proc = 'bfadv'

    if system == 'cartesius':
        outfile = '/scratch-local/rfischer/Kooi_data/data_output/allrho/res_'+res+'/allr/regional_'+region+'_'+proc+'_'+s+'_'+yr+'_3D_grid'+res+'_allrho_allr_'+str(round(simdays,2))+'days_'+str(secsdt)+'dtsecs_'+str(round(hrsoutdt,2))+'hrsoutdt' 
    elif system == 'gemini':
         outfile = '/scratch/rfischer/Kooi_data/data_output/regional_'+region+'_'+proc+'_'+s+'_'+yr+'_0'+str(mortality_rate)[2:]+'mort_'+mixing+'mixing_'+str(round(simdays,2))+'days_'+str(secsdt)+'dtsecs_'+str(round(hrsoutdt,2))+'hrsoutdt'

    pfile= ParticleFile(outfile, pset, outputdt=delta(hours = hrsoutdt))

    pset.execute(kernels, runtime=delta(days=simdays), dt=delta(seconds = secsdt), output_file=pfile, verbose_progress=True, recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle, ErrorCode.ErrorInterpolation: DeleteParticle})

    pfile.close()

    print('Execution finished')



