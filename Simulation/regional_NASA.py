from parcels import FieldSet, ParticleSet, JITParticle, ErrorCode, ParticleFile, Variable, Field, ParcelsRandom
from parcels.application_kernels.TEOSseawaterdensity import PolyTEOS10_bsq
from datetime import timedelta as delta
import numpy as np
from glob import glob
import xarray as xr
import math

seed = 123
ParcelsRandom.seed(seed)

#------ Choose ------:
simdays = 90
secsdt = 60 #30
hrsoutdt = 6
region = 'EqPac'
yr = '2004'
mon ='01'

class plastic_particle(JITParticle): #ScipyParticle): #
    u = Variable('u', dtype=np.float32,to_write=False)
    v = Variable('v', dtype=np.float32,to_write=False)
    w = Variable('w', dtype=np.float32,to_write=False)
    w_adv = Variable('w_adv', dtype=np.float32,to_write=False)
    temp = Variable('temp',dtype=np.float32,to_write=False)
    density = Variable('density',dtype=np.float32,to_write=False)
    mld = Variable('mld',dtype=np.float32,to_write=True)
    w_m = Variable('w_m',dtype=np.float32,to_write=True)
    vs_init = Variable('vs_init',dtype=np.float32,to_write=False)
    vs = Variable('vs',dtype=np.float32,to_write=False)
    r_pl = Variable('r_pl',dtype=np.float32,to_write='once')
    rho_pl = Variable('rho_pl',dtype=np.float32,to_write='once')
    rho_tot = Variable('rho_tot',dtype=np.float32,to_write=False)
    r_tot = Variable('r_tot',dtype=np.float32,to_write=True)
    delta_rho = Variable('delta_rho',dtype=np.float32,to_write=False)
    kin_visc = Variable('kin_visc',dtype=np.float32,to_write=False)
    sw_visc = Variable('sw_visc',dtype=np.float32,to_write=False)
    a = Variable('a',dtype=np.float32,to_write=True)
    rho_bf = Variable('rho_bf',dtype=np.float32,to_write=False)
    d_phy = Variable('d_phy',dtype=np.float32,to_write=False)
    Gro = Variable('Gro',dtype=np.float32,to_write=False)
    Graz = Variable('Graz',dtype=np.float32,to_write=False)
    Sen = Variable('Sen',dtype=np.float32,to_write=False)
    Resp = Variable('Resp',dtype=np.float32,to_write=False)
    Doc = Variable('Doc',dtype=np.float32,to_write=False)
    a_coll = Variable('a_coll', dtype=np.float32, to_write=True)
    a_gro = Variable('a_gro',dtype=np.float32,to_write=True)
    a_graz = Variable('a_graz',dtype=np.float32,to_write=True)
    a_sen = Variable('a_sen',dtype=np.float32,to_write=True)
    a_resp = Variable('a_resp',dtype=np.float32,to_write=True)
    a_doc = Variable('a_doc',dtype=np.float32,to_write=True)

def AdvectionRK4_3D(particle, fieldset, time):
    """Advection of particles using fourth-order Runge-Kutta integration including vertical velocity.
    Function needs to be converted to Kernel object before execution"""
    (u1, v1, w1) = fieldset.UVW[particle]
    lon1 = particle.lon + u1*.5*particle.dt
    lat1 = particle.lat + v1*.5*particle.dt
    dep1 = particle.depth + w1*.5*particle.dt
    if dep1 > 5.:
        (u2, v2, w2) = fieldset.UVW[time + .5 * particle.dt, dep1, lat1, lon1, particle]
        lon2 = particle.lon + u2*.5*particle.dt
        lat2 = particle.lat + v2*.5*particle.dt
        dep2 = particle.depth + w2*.5*particle.dt
        if dep2 > 5.:
            (u3, v3, w3) = fieldset.UVW[time + .5 * particle.dt, dep2, lat2, lon2, particle]
            lon3 = particle.lon + u3*particle.dt
            lat3 = particle.lat + v3*particle.dt
            dep3 = particle.depth + w3*particle.dt
            if dep2 >5.:
                (u4, v4, w4) = fieldset.UVW[time + particle.dt, dep3, lat3, lon3, particle]
                if particle.depth + (w1 + 2*w2 + 2*w3 + w4) / 6. * particle.dt > 5.:
                    particle.depth += (w1 + 2*w2 + 2*w3 + w4) / 6. * particle.dt
                else:
                    particle.depth = 5.
            else:
                particle.depth = 5.
        else:particle.depth = 5.
    else:
        particle.depth = 5.
    
    particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
    particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt
    
    
def Profiles(particle, fieldset, time):
    particle.temp = fieldset.cons_temperature[time, particle.depth,particle.lat,particle.lon]
    particle.d_phy= fieldset.d_phy[time, particle.depth,particle.lat,particle.lon]
    particle.Gro = fieldset.gro[time, particle.depth,particle.lat,particle.lon]
    particle.Graz = fieldset.graz[time, particle.depth,particle.lat,particle.lon]
    particle.Sen = fieldset.sen[time, particle.depth,particle.lat,particle.lon]
    particle.Resp = fieldset.resp[time, particle.depth,particle.lat,particle.lon]
    particle.Doc = fieldset.DOC[time, particle.depth,particle.lat,particle.lon]

    mu_w = 4.2844E-5 + (1/((0.157*(particle.temp + 64.993)**2)-91.296))
    A = 1.541 + 1.998E-2*particle.temp - 9.52E-5*particle.temp**2
    B = 7.974 - 7.561E-2*particle.temp + 4.724E-4*particle.temp**2
    S_sw = fieldset.abs_salinity[time, particle.depth, particle.lat, particle.lon]/1000
    particle.sw_visc = mu_w*(1 + A*S_sw + B*S_sw**2)
    particle.kin_visc = particle.sw_visc/particle.density
    particle.w_adv = fieldset.W[time,particle.depth,particle.lat,particle.lon]
    mld = fieldset.mldep[time, particle.depth, particle.lat, particle.lon]
    particle.mld = particle.depth/mld

def uniform_release(n_locs, n_particles_per_bin, n_bins, e_max=-3, e_min=-5):
    '''
    Create a set of particle radii with a fixed amount of particles per bin for a given number of release locations.
    The bins are spaced logarithmically.

    :param n_locs: number of release locations
    :param n_particles_per_bin: number of particles per bin:
    :param n_bins: number of bins between 1E-e_max and 1E-e_min
    :param e_max: Exponent of the largest particle. -3 -> 1E-3 m = 1 mm
    :param e_min: Exponent of the smallest particle. -6 -> 1E-6 = 1 um
    '''
    sizes = np.logspace(e_min, e_max, n_bins)
    location_r_pls = np.repeat(sizes, n_particles_per_bin)
    r_pls = np.tile(location_r_pls, [n_locs,1])
    return r_pls

def DeleteParticle(particle, fieldset, time):
    """Kernel for deleting particles if they are out of bounds."""
    print('particle is deleted out of bounds at lon = '+str(particle.lon)+', lat ='+str(particle.lat)+', depth ='+str(particle.depth))
    particle.delete()
    
def DeleteSurface(particle, fieldset, time):
    """Kernel for deleting particles if they are out of bounds."""
    print('particle is deleted through surface at depth = '+str(particle.depth))
    particle.delete()

def DeleteParticleInterp(particle, fieldset, time):
    """Kernel for deleting particles if they are out of bounds."""
    print('particle is deleted due to an interpolation error at lon = '+str(particle.lon)+', lat ='+str(particle.lat)+', depth ='+str(particle.depth))
    particle.delete()

def markov_0_KPP_reflect(particle, fieldset, time):
    """
    If a particle tries to cross the boundary, then it is reflected back
    Author: Victor Onink
    Adapted 1D -> 3D
    Adapted for NASA_GISS modelE using 'kvert'
    """
    K_z = fieldset.kz[time, particle.depth, particle.lat, particle.lon]
    dK_z_p = fieldset.dkz[time, particle.depth, particle.lat, particle.lon]

    # According to Ross & Sharples (2004), first the deterministic part of equation 1
    deterministic = dK_z_p * particle.dt

    # The random walk component
    R = ParcelsRandom.uniform(-1., 1.) * math.sqrt(math.fabs(particle.dt) * 3)
    bz = math.sqrt(2 * K_z)

    # Total movement
    w_m_step = deterministic + R * bz
    particle.w_m = w_m_step/particle.dt

    # The ocean surface acts as a lid off of which the plastic bounces if tries to cross the ocean surface
    potential = particle.depth + w_m_step
    if potential < 5.:
        particle.depth = 5. + (5. - potential)
    else:
        particle.depth = potential

def NASA_GISS(particle,fieldset,time):
    """
    Kernel to compute the vertical velocity (Vs) of particles due to changes in ambient algal concentrations, growth and death of attached algae based on Kooi et al. 2017 model
    """
    # ------ Constants and algal properties -----
    g = fieldset.G            # gravitational acceleration [m s-2]
    k = fieldset.K            # Boltzmann constant [m2 kg d-2 K-1] now [s-2] (=1.3804E-23)
    rho_fr = fieldset.Rho_fr  # frustule density [g m-3]
    v_a = fieldset.V_a        # Volume of 1 algal cell [m-3]
    a_diss = fieldset.Diss    # dissolution rate [s-1]
    gamma = fieldset.Gamma    # shear [s-1]

    # ------ Profiles from MEDUSA or Kooi theoretical profiles -----
    z = particle.depth  # [m]
    t = particle.temp  # [oC]
    sw_visc = particle.sw_visc  # [kg m-1 s-1]
    kin_visc = particle.kin_visc  # [m2 s-1]
    rho_sw = particle.density  # [kg m-3]
    a = particle.a  # [no. m-2]
    vs = particle.vs  # [m s-1]

    #------ Nitrogen to cell ratios for ambient algal concentrations ('aa') and algal growth ('mu_aa') from NEMO output (no longer using N:C:AA (Redfield ratio), directly N:AA from Menden-Deuer and Lessard 2000)
    med_N2cell = 356.04e-09 # [mgN cell-1] median value is used below (as done in Kooi et al. 2017)
    med_chlcell = 1e-08     # [mgChl cell-1] 
    wt_N = fieldset.Wt_N    # atomic weight of 1 mol of N = 14.007 g
    wt_Si = fieldset.Wt_Si  # atomic weight of 1 mol of Si = 28.0855 g

    #------ Ambient algal concentration from MEDUSA's non-diatom + diatom phytoplankton
    d = particle.d_phy                  # [mg chl m-3] diatom concentration that attaches to plastic particles
    
    d2 = d/med_N2cell                   # conversion from [mg chl m-3] to [no. m-3]

    ad = d2                            # [no m-3] ambient diatoms
    
    #------ Primary productivity (algal growth) from MEDUSA TPP3 (no longer condition of only above euphotic zone, since not much diff in results)
    a_gro = particle.Gro              # [mg chl m-3 s-1]
    mu_n = a_gro/med_chlcell          # conversion from [mg chl m-3 s-1] to [no. m-3 s-1]
    if ad>0:
        mu_n2 = mu_n/ad                   # conversion from [no. m-3 s-1] to [s-1]
    else:
        mu_n2 = 0.

    if mu_n2<0.:
        mu_ad = 0.
    elif mu_n2>1.85:
        mu_ad = 1.85           # [s-1] maximum growth rate
    else:
        mu_ad = mu_n2          # [s-1]

    #------ Grazing -----
    gr0 = particle.Graz         # [mg chl m-3 s-1]
    gr_n = gr0/med_chlcell      # conversion to [no. m-3 s-1]
    if ad>0:
        gr_ad = gr_n/ad             # conversion to [s-1]
    else:
        gr_ad = 0.

    #------ Senesence -----
    s0 = particle.Sen          # [mg chl m-3 s-1]
    s_n = s0/med_chlcell       # conversion to [no. m-3 s-1]
    if ad>0:
        s_ad = s_n/ad              # conversion to [s-1]
    else:
        s_ad=0.

    #------ Respiration -----
    resp0 = particle.Resp         # [mg chl m-3 s-1]
    resp_n = resp0/med_chlcell    # conversion to [no. m-3 s-1]
    if ad>0:
        resp_ad = resp_n/ad           # conversion to [s-1]
    else:
        resp_ad = 0.

    #------ DOC -----
    doc0 = particle.Doc         # [mg chl m-3 s-1]
    doc_n = doc0/med_chlcell    # conversion to [no. m-3 s-1]
    if ad>0:
        doc_ad = doc_n/ad           # conversion to [s-1]
    else:
        doc_ad = 0.

    #------ Density -----
    rho_bf = fieldset.Rho_bf
    particle.rho_bf = rho_bf

    #------ Volumes -----
    v_pl = (4./3.)*math.pi*particle.r_pl**3.             # volume of plastic [m3]
    theta_pl = 4.*math.pi*particle.r_pl**2.              # surface area of plastic particle [m2]
    r_a = ((3./4.)*(v_a/math.pi))**(1./3.)               # radius of an algal cell [m]

    v_bfa = (v_a*a)*theta_pl                             # volume of living biofilm [m3]

    v_tot = v_bfa + v_pl                                 # volume of total [m3]
    t_bf = ((v_tot*(3./(4.*math.pi)))**(1./3.))-particle.r_pl  # biofilm thickness [m]

    #------ Diffusivity -----
    r_tot = particle.r_pl + t_bf                              # total radius [m]
    rho_tot = (v_pl * particle.rho_pl + v_bfa*rho_bf)/v_tot   # total density [kg m-3] dead cells = frustule
    theta_tot = 4.*math.pi*r_tot**2.                          # surface area of total [m2]
    d_pl = k * (t + 273.16)/(6. * math.pi * sw_visc * r_tot)  # diffusivity of plastic particle [m2 s-1]
    d_a = k * (t + 273.16)/(6. * math.pi * sw_visc * r_a)     # diffusivity of algal cells [m2 s-1]

    #------ Encounter rates -----
    beta_abrown = 4.*math.pi*(d_pl + d_a)*(r_tot + r_a)       # Brownian motion [m3 s-1]
    beta_ashear = 1.3*gamma*((r_tot + r_a)**3.)               # advective shear [m3 s-1]
    beta_aset = (1./2.)*math.pi*r_tot**2. * abs(vs)           # differential settling [m3 s-1]
    beta_a = beta_abrown + beta_ashear + beta_aset            # collision rate [m3 s-1]

    #------ Attached algal growth (Eq. 11 in Kooi et al. 2017) -----
    a_coll = (beta_a*ad)/theta_pl*fieldset.collision_eff      # [no. m-2 s-1] collisions with diatoms
    a_growth = mu_ad*a                                        # [no. m-2 s-1]

    a_grazing = gr_ad*a                                       # grazing losses [no. m-2 s-1]
    a_senesence = s_ad*a                                      # senesence losses [no. m-2 s-1] 
    a_respiration = resp_ad*a                                 # non-linear losses [no. m-2 s-1]
    a_DOCloss = doc_ad*a                                      # DOC losses [no. m-2 s-1]

    particle.a_coll = a_coll
    particle.a_gro = a_growth
    particle.a_graz = a_grazing
    particle.a_sen = a_senesence
    particle.a_resp = a_respiration
    particle.a_doc = a_DOCloss
    particle.a += (a_coll + a_growth + a_grazing + a_senesence + a_respiration + a_DOCloss) * particle.dt

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
    if delta_rho > 0: # sinks
        vs = (g * kin_visc * w * delta_rho)**(1./3.)
    else: #rises
        a_del_rho = delta_rho*-1.
        vs = -1.*(g * kin_visc * w * a_del_rho)**(1./3.)  # m s-1

    particle.vs_init = vs

    z0 = z + vs * particle.dt
    if z0 <=5. or z0 >= 4000.: # NEMO's 'surface depth'
        vs = 0
        particle.depth = 5.
    else:
        particle.depth += vs * particle.dt

    particle.vs = vs
    particle.rho_tot = rho_tot
    particle.r_tot = r_tot
    particle.delta_rho = delta_rho

#------ Fieldset grid  ------
if region == 'NPSG':
    minlat = 20
    maxlat = 45
    minlon = 110 # -180 #75
    maxlon = -120 #45
elif region == 'EqPac':
    minlat = -20
    maxlat = 20
    minlon = 160
    maxlon = -120
elif region == 'SO':
    minlat = -75
    maxlat = -45
    minlon = -15
    maxlon = 25

""" Defining the fieldset"""
dirread = '/data/oceanparcels/input_data/NASA_GISS/modelE/'
dkdirread = '/data/oceanparcels/input_data/NASA_GISS/dKz/'

yr0 = yr
ufiles = sorted(glob(dirread+yr+'*_oijlh*.nc'))
bgcfiles = sorted(glob(dirread+yr+'*_obijlh*.nc'))
surffiles = sorted(glob(dirread+yr+'*_oijh*.nc'))
dKzfiles = sorted(glob(dkdirread+yr+'*'))

# mesh_mask = dirread_mesh+'coordinates.nc'
filenames = {'U': {'lon': ufiles[0], 'lat': ufiles[0], 'depth': ufiles[0], 'data': ufiles}, #'depth': wfiles,
             'V': {'lon': ufiles[0], 'lat': ufiles[0], 'depth': ufiles[0], 'data': ufiles},
             'W': {'lon': ufiles[0], 'lat': ufiles[0], 'depth': ufiles[0], 'data': ufiles},
             'kz': {'lon': ufiles[0], 'lat': ufiles[0], 'depth': ufiles[0], 'data': ufiles},
             'dkz': {'lon': dKzfiles[0], 'lat': dKzfiles[0], 'depth': dKzfiles[0], 'data': dKzfiles},
             'abs_salinity': {'lon': ufiles[0], 'lat': ufiles[0], 'depth': ufiles[0], 'data': ufiles},
             'cons_temperature': {'lon': ufiles[0], 'lat': ufiles[0], 'depth': ufiles[0], 'data': ufiles},
             'd_phy': {'lon': bgcfiles[0], 'lat': bgcfiles[0], 'depth': bgcfiles[0], 'data': bgcfiles},       # mg chl/m3
             'sen': {'lon': bgcfiles[0], 'lat': bgcfiles[0], 'depth': bgcfiles[0], 'data': bgcfiles},         # mg chl/m3/s Senesence
             'resp': {'lon': bgcfiles[0], 'lat': bgcfiles[0], 'depth': bgcfiles[0], 'data': bgcfiles},        # mg chl/m3/s Respiration
             'DOC': {'lon': bgcfiles[0], 'lat': bgcfiles[0], 'depth': bgcfiles[0], 'data': bgcfiles},         # mg chl/m3/s Loss to DOC
             'graz': {'lon': bgcfiles[0], 'lat': bgcfiles[0], 'depth': bgcfiles[0], 'data': bgcfiles},        # mg chl/m3/s Grazing
             'gro': {'lon': bgcfiles[0], 'lat': bgcfiles[0], 'depth': bgcfiles[0], 'data': bgcfiles},         # mg chl/m3/s Growth
             'sink': {'lon': bgcfiles[0], 'lat': bgcfiles[0], 'depth': bgcfiles[0], 'data': bgcfiles},        # mg chl/m3/s
             'mldep': {'lon': ufiles[0], 'lat': ufiles[0],  'data': surffiles}}

variables = {'U': 'u',
             'V': 'v',
             'W': 'w',
             'kz': 'kvert',
             'dkz': 'dKz',
             'abs_salinity': 'salt',
             'cons_temperature': 'temp',
             'd_phy': 'diat',             # mg chl/m3
             'sen': 'diatdead',         # mg chl/m3/s Senesence
             'resp': 'diatDIC',         # mg chl/m3/s Respiration
             'DOC': 'diatDOC',          # mg chl/m3/s Loss to DOC
             'graz': 'diatgraz',        # mg chl/m3/s Grazing
             'gro': 'diatgro',          # mg chl/m3/s Growth
             'sink': 'diatsink',        # mg chl/m3/s
             'mldep': 'mldep'}

dimensions = {'U': {'lon': 'lono', 'lat': 'lato', 'depth': 'zoc', 'time': 'time'}, #time_centered
              'V': {'lon': 'lono', 'lat': 'lato', 'depth': 'zoc', 'time': 'time'},
              'W': {'lon': 'lono', 'lat': 'lato', 'depth': 'zoc', 'time': 'time'},
              'kz': {'lon': 'lono', 'lat': 'lato', 'depth': 'zoc', 'time': 'time'},
              'dkz': {'lon': 'longitude', 'lat': 'latitude', 'depth': 'depth', 'time': 'time'},
              'abs_salinity': {'lon': 'lono', 'lat': 'lato', 'depth': 'zoc', 'time': 'time'},
              'cons_temperature': {'lon': 'lono', 'lat': 'lato', 'depth': 'zoc', 'time': 'time'},
              'd_phy': {'lon': 'lono', 'lat': 'lato', 'depth': 'zoc', 'time': 'time'},             # mg chl/m3
              'sen': {'lon': 'lono', 'lat': 'lato', 'depth': 'zoc', 'time': 'time'},         # mg chl/m3/s Senesence
              'resp': {'lon': 'lono', 'lat': 'lato', 'depth': 'zoc', 'time': 'time'},         # mg chl/m3/s Respiration
              'DOC': {'lon': 'lono', 'lat': 'lato', 'depth': 'zoc', 'time': 'time'},          # mg chl/m3/s Loss to DOC
              'graz': {'lon': 'lono', 'lat': 'lato', 'depth': 'zoc', 'time': 'time'},        # mg chl/m3/s Grazing
              'gro': {'lon': 'lono', 'lat': 'lato', 'depth': 'zoc', 'time': 'time'},          # mg chl/m3/s Growth
              'sink': {'lon': 'lono', 'lat': 'lato', 'depth': 'zoc', 'time': 'time'},        # mg chl/m3/s
              'mldep': {'lon': 'lono', 'lat': 'lato', 'time': 'time'}}

fieldset = FieldSet.from_nemo(filenames, variables, dimensions)

# ------ Defining constants ------
fieldset.add_constant('Gr_a', 0.39 / 86400.)
fieldset.add_constant('collision_eff', 1.)
fieldset.add_constant('K', 1.0306E-13 / (86400. ** 2.))  # Boltzmann constant [m2 kg d-2 K-1] now [s-2] (=1.3804E-23)
fieldset.add_constant('Rho_bf', 1388.)                   # density of biofilm [g m-3]
fieldset.add_constant('Rho_fr', 2200.)                   # density of frustule [g m-3] median value from Miklasz & Denny 2010
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

# ------ markov constants ------
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
    lat_release0 = np.tile(np.linspace(28,36,n_res),[n_res,1])
    lon_release0 = np.tile(np.linspace(-135,-143,n_res),[n_res,1])

    lat_release = np.tile(lat_release0.T, [n_sizebins*n_particles_per_bin,1,1]).T
    lon_release = np.tile(lon_release0, [n_sizebins*n_particles_per_bin,1,1]).T
elif region == 'EqPac':
    lat_release0 = np.tile(np.linspace(-4,4,n_res),[n_res,1])
    lon_release0 = np.tile(np.linspace(-140,-148,n_res),[n_res,1])

    lat_release = np.tile(lat_release0.T, [n_sizebins*n_particles_per_bin,1,1]).T
    lon_release = np.tile(lon_release0, [n_sizebins*n_particles_per_bin,1,1]).T
elif region == 'SO':
    lat_release0  = np.tile(np.linspace(-65,-55,n_res),[n_res,1])
    lon_release0 = np.tile(np.linspace(-10,0,n_res),[n_res,1])

    lat_release = np.tile(lat_release0.T, [n_sizebins*n_particles_per_bin,1,1]).T
    lon_release = np.tile(lon_release0, [n_sizebins*n_particles_per_bin,1,1]).T

z_release = np.tile(fieldset.U.grid.depth[0]+1,[n_res,n_res, n_sizebins*n_particles_per_bin])
res = '1x1'

rho_pls = np.tile(920, [n_res, n_res, n_sizebins*n_particles_per_bin])
r_pls = uniform_release(n_locs, n_particles_per_bin, n_sizebins)

pset = ParticleSet.from_list(fieldset=fieldset,         # the fields on which the particles are advected
                             pclass=plastic_particle,   # the type of particles (JITParticle or ScipyParticle)
                             lon= lon_release, #-160.,  # a vector of release longitudes
                             lat= lat_release, #36.,
                             time = 0,
                             depth = z_release,
                             lonlatdepth_dtype=np.float64,
                             r_pl = r_pls,
                             rho_pl = rho_pls,
                             r_tot = r_pls,
                             rho_tot = rho_pls)

outfile = '/scratch/rfischer/Kooi_data/data_output/regional_'+region+'_NASA_'+str(simdays)+'simdays_'+str(hrsoutdt)+'hrsoutdt'

pfile= ParticleFile(outfile, pset, outputdt=delta(hours = hrsoutdt))

pset.execute(pset.Kernel(PolyTEOS10_bsq)+pset.Kernel(Profiles) + pset.Kernel(AdvectionRK4_3D)+pset.Kernel(markov_0_KPP_reflect)+pset.Kernel(NASA_GISS), runtime=delta(days=simdays), dt=delta(seconds = secsdt), output_file=pfile, verbose_progress=True, recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle, ErrorCode.ErrorThroughSurface: DeleteSurface, ErrorCode.ErrorInterpolation: DeleteParticleInterp})

pfile.close()
