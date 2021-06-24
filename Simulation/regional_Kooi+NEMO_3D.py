"""
Authors: Delphine Lobelle, Reint Fischer

Executable python script to simulate regional biofouling particles with parameterized wind and tidal mixing.
"""

from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4_3D, ErrorCode, ParticleFile, Variable, Field, ParcelsRandom 
from parcels.application_kernels.TEOSseawaterdensity import PolyTEOS10_bsq
from datetime import timedelta as delta
import numpy as np
from numpy.random import default_rng
from glob import glob
import xarray as xr
import warnings
import math 
from argparse import ArgumentParser
warnings.filterwarnings("ignore")

seed = 123
ParcelsRandom.seed(seed)
rng = default_rng(seed)

#------ Choose ------:
simdays = 458
secsdt = 60 #30
hrsoutdt = 12

"""functions and kernels"""

def MEDUSA_full_grazing(particle,fieldset,time):
    """
    Kernel to compute the vertical velocity (Vs) of particles due to changes in ambient algal concentrations, growth and death of attached algae based on Kooi et al. 2017 model settling velocity and MEDUSA 2.0 biofilm dynamics, including modelling of the 3D mesozooplankton grazing of diatoms
    """
    # ------ Constants and algal properties -----
    g = fieldset.G            # gravitational acceleration [m s-2]
    k = fieldset.K            # Boltzmann constant [m2 kg d-2 K-1] now [s-2] (=1.3804E-23)
    rho_fr = fieldset.Rho_fr  # frustule density [g m-3]
    rho_cy = fieldset.Rho_cy  # cytoplasm density [g m-3]
    v_a = fieldset.V_a        # Volume of 1 algal cell [m-3]
    r20 = fieldset.R20        # respiration rate [s-1]
    q10 = fieldset.Q10        # temperature coefficient respiration [-]
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
    wt_N = fieldset.Wt_N    # atomic weight of 1 mol of N = 14.007 g
    wt_Si = fieldset.Wt_Si  # atomic weight of 1 mor of Si = 28.0855

    #------ Ambient algal concentration from MEDUSA's non-diatom + diatom phytoplankton
    n0 = particle.nd_phy+particle.d_phy # [mmol N m-3] total plankton concentration engaging in primary production in MEDUSA
    d0 = particle.d_phy                 # [mmol N m-3] diatom concentration that attaches to plastic particles

    n = n0*wt_N                         # conversion from [mmol N m-3] to [mg N m-3]
    d = d0*wt_N                         # conversion from [mmol N m-3] to [mg N m-3]

    n2 = n/med_N2cell                   # conversion from [mg N m-3] to [no. m-3]
    d2 = d/med_N2cell                   # conversion from [mg N m-3] to [no. m-3]

    if n2<0.:
        aa = 0.
    else:
        aa = n2                        # [no m-3] ambient algae - to compare to Kooi model
    ad = d2                            # [no m-3] ambient diatoms

    #------ Primary productivity (algal growth) from MEDUSA TPP3
    tpp0 = particle.tpp3              # [mmol N m-3 d-1]
    mu_n0 = tpp0*wt_N                 # conversion from [mmol N m-3 d-1] to [mg N m-3 d-1] (atomic weight of 1 mol of N = 14.007 g)
    mu_n = mu_n0/med_N2cell           # conversion from [mg N m-3 d-1] to [no. m-3 d-1]
    if aa>0:                          # If there are any ambient algae
        mu_n2 = mu_n/aa               # conversion from [no. m-3 d-1] to [d-1]
    else:
        mu_n2 = 0.

    if mu_n2<0.:
        mu_aa = 0.
    elif mu_n2>1.85:
        mu_aa = 1.85/86400.           # maximum growth rate
    else:
        mu_aa = mu_n2/86400.          # conversion from d-1 to s-1

    #------ Grazing -----
    # Based on equations 54 and 55 in Yool et al. 2013
    FPn = fieldset.pmPn * math.pow(particle.nd_phy,2)     # (mmol N m-3)**2 Interest in available non-diatoms
    FPd = fieldset.pmPd * math.pow(particle.d_phy,2)      # (mmol N m-3)**2 Interest in available diatoms
    FZmu = fieldset.pmZmu * math.pow(particle.mic_zoo,2)  # (mmol N m-3)**2 Interest in available microzooplankton
    FD = fieldset.pmD * math.pow(particle.detr,2)         # (mmol N m-3)**2 Interest in available detritus
    Fm = FPn + FPd + FZmu + FD                            # (mmol N m-3)**2 Interest in total available food

    GmPd = (fieldset.Gm * fieldset.pmPd * math.pow(particle.d_phy,2) * particle.mes_zoo)/(math.pow(fieldset.km,2) + Fm)  # [mmol N m-3 s-1]

    gr0 = GmPd
    gr1 = gr0*wt_N            # conversion to [mg N m-3 s-1]
    gr_n = gr1/med_N2cell     # conversion to [no. m-3 s-1]
    gr_ad = gr_n/ad           # conversion to [s-1]

    #------ Non-linear losses ------
    a_nlin0 = fieldset.mu2*particle.d_phy*particle.d_phy/(fieldset.kPd+particle.d_phy)  # ambient diatom non-linear losses [mmol N m-3 s-1]
    a_nlin1 = a_nlin*wt_N                           # conversion to [mg N m-3 s-1]
    a_nlin_n = a_nlin1/med_N2cell                   # conversion to [no. m-3 s-1]
    a_nlin = a_nlin_n/ad                            # conversion to [s-1]

    #------ N:Si ratio density ------
    R_Si_N = particle.d_si/particle.d_phy  # [(mmol N) (mmol Si)-1]

    particle.Si_N = R_Si_N

    rho_bf = fieldset.Rho_bf
    particle.rho_bf = rho_bf

    #------ Volumes -----
    v_pl = (4./3.)*math.pi*particle.r_pl**3.             # volume of plastic [m3]
    theta_pl = 4.*math.pi*particle.r_pl**2.              # surface area of plastic particle [m2]
    r_a = ((3./4.)*(v_a/math.pi))**(1./3.)               # radius of an algal cell [m]

    v_bfa = (v_a*a)*theta_pl                              # volume of living biofilm [m3]
    v_tot = v_bfa + v_pl                                  # volume of total (biofilm + plastic) [m3]
    t_bf = ((v_tot*(3./(4.*math.pi)))**(1./3.))-particle.r_pl  # biofilm thickness [m]

    #------ Diffusivity -----
    r_tot = particle.r_pl + t_bf                              # total radius [m]
    rho_tot = (v_pl * particle.rho_pl + v_bfa*rho_bf)/v_tot   # total density [kg m-3]
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
    a_growth = mu_aa*a

    a_grazing = gr_ad*a
    a_linear = fieldset.mu1*a                                 # linear losses [no. m-2 s-1]
    a_non_linear = a_nlin*a                                   # non-linear losses [no. m-2 s-1]
    a_resp = (q10**((t-20.)/10.))*r20*a                       # [no. m-2 s-1] respiration

    particle.a_coll = a_coll
    particle.a_growth = a_growth

    particle.a_gr = a_grazing
    particle.a_nl = a_non_linear
    particle.a_l = a_linear
    particle.a_resp = a_resp
    particle.a += (a_coll + a_growth - a_grazing - a_resp - a_non_linear) * particle.dt

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
    if z0 <=0.6 or z0 >= 4000.: # NEMO's 'surface depth'
        vs = 0
        particle.depth = 0.6
    else:
        particle.depth += vs * particle.dt

    particle.vs = vs
    particle.rho_tot = rho_tot
    particle.r_tot = r_tot
    particle.delta_rho = delta_rho
    particle.t_bf = t_bf          

def MEDUSA_detritus_full_grazing(particle,fieldset,time):
    """
    Kernel to compute the vertical velocity (Vs) of particles due to changes in ambient algal concentrations, growth and death of attached algae based on Kooi et al. 2017 model
    """
    # ------ Constants and algal properties -----
    g = fieldset.G            # gravitational acceleration [m s-2]
    k = fieldset.K            # Boltzmann constant [m2 kg d-2 K-1] now [s-2] (=1.3804E-23)
    rho_fr = fieldset.Rho_fr  # frustule density [g m-3]
    rho_cy = fieldset.Rho_cy  # cytoplasm density [g m-3]
    v_a = fieldset.V_a        # Volume of 1 algal cell [m-3]
    a_diss = fieldset.Diss    # dissolution rate [s-1]
    r20 = fieldset.R20        # respiration rate [s-1]
    q10 = fieldset.Q10        # temperature coefficient respiration [-]
    gamma = fieldset.Gamma    # shear [s-1]

    # ------ Profiles from MEDUSA or Kooi theoretical profiles -----
    z = particle.depth  # [m]
    t = particle.temp  # [oC]
    sw_visc = particle.sw_visc  # [kg m-1 s-1]
    kin_visc = particle.kin_visc  # [m2 s-1]
    rho_sw = particle.density  # [kg m-3]
    a = particle.a  # [no. m-2]
    a_dead = particle.a_dead
    vs = particle.vs  # [m s-1]

    #------ Nitrogen to cell ratios for ambient algal concentrations ('aa') and algal growth ('mu_aa') from NEMO output (no longer using N:C:AA (Redfield ratio), directly N:AA from Menden-Deuer and Lessard 2000)
    med_N2cell = 356.04e-09 # [mgN cell-1] median value is used below (as done in Kooi et al. 2017)
    wt_N = fieldset.Wt_N    # atomic weight of 1 mol of N = 14.007 g
    wt_Si = fieldset.Wt_Si  # atomic weight of 1 mol of Si = 28.0855 g

    #------ Ambient algal concentration from MEDUSA's non-diatom + diatom phytoplankton
    n0 = particle.nd_phy+particle.d_phy # [mmol N m-3] total plankton concentration engaging in primary production in MEDUSA
    d0 = particle.d_phy                 # [mmol N m-3] diatom concentration that attaches to plastic particles

    n = n0*wt_N                         # conversion from [mmol N m-3] to [mg N m-3]
    d = d0*wt_N                         # conversion from [mmol N m-3] to [mg N m-3]

    n2 = n/med_N2cell                   # conversion from [mg N m-3] to [no. m-3]
    d2 = d/med_N2cell                   # conversion from [mg N m-3] to [no. m-3]

    if n2<0.:
        aa = 0.
    else:
        aa = n2                        # [no m-3] ambient algae - to compare to Kooi model
    ad = d2                            # [no m-3] ambient diatoms

    #------ Primary productivity (algal growth) from MEDUSA TPP3 (no longer condition of only above euphotic zone, since not much diff in results)
    tpp0 = particle.tpp3              # [mmol N m-3 d-1]
    mu_n0 = tpp0*wt_N                 # conversion from [mmol N m-3 d-1] to [mg N m-3 d-1] (atomic weight of 1 mol of N = 14.007 g)
    mu_n = mu_n0/med_N2cell           # conversion from [mg N m-3 d-1] to [no. m-3 d-1]
    if aa>0:
        mu_n2 = mu_n/aa                   # conversion from [no. m-3 d-1] to [d-1]
    else:
        mu_n2=0.

    if mu_n2<0.:
        mu_aa = 0.
    elif mu_n2>1.85:
        mu_aa = 1.85/86400.           # maximum growth rate
    else:
        mu_aa = mu_n2/86400.          # conversion from d-1 to s-1

    #------ Grazing -----
    # Based on equations 54 and 55 in Yool et al. 2013
    FPn = fieldset.pmPn * math.pow(particle.nd_phy,2)         # (mmol N m-3)**2 Interest in available non-diatoms
    FPd = fieldset.pmPd * math.pow(particle.d_phy,2)          # (mmol N m-3)**2 Interest in available diatoms
    FZmu = fieldset.pmZmu * math.pow(particle.mic_zoo,2)      # (mmol N m-3)**2 Interest in available microzooplankton
    FD = fieldset.pmD * math.pow(particle.detr,2)             # (mmol N m-3)**2 Interest in available detritus
    Fm = FPn + FPd + FZmu + FD                                # (mmol N m-3)**2 Interest in total available food

    GmPd = (fieldset.Gm * fieldset.pmPd * math.pow(particle.d_phy,2) * particle.mes_zoo)/(fieldset.km + Fm)  # [mmol N m-3 s-1]

    gr0 = GmPd
    gr1 = gr0*wt_N              # conversion to [mg N m-3 s-1]
    gr_n = gr1/med_N2cell       # conversion to [no. m-3 s-1]
    if ad>0:
        gr_ad = gr_n/ad             # conversion to [s-1]
    else:
        gr_ad = 0.

    #------ Non-linear losses ------
    a_nlin0 = fieldset.mu2*particle.d_phy*particle.d_phy/(fieldset.kPd+particle.d_phy)  # ambient diatom non-linear losses [mmol N m-3 s-1]
    a_nlin1 = a_nlin*wt_N                           # conversion to [mg N m-3 s-1]
    a_nlin_n = a_nlin1/med_N2cell                   # conversion to [no. m-3 s-1]
    if ad>0:
        a_nlin = a_nlin_n/ad                            # conversion to [s-1]
    else:
        a_nlin = 0.

    #------ N:Si ratio density ------
    R_Si_N = particle.d_si/particle.d_phy  # [(mmol N)-1 (mmol Si)]

    particle.Si_N = R_Si_N

    rho_bf = fieldset.Rho_bf
    particle.rho_bf = rho_bf

    #------ Volumes -----
    v_pl = (4./3.)*math.pi*particle.r_pl**3.             # volume of plastic [m3]
    theta_pl = 4.*math.pi*particle.r_pl**2.              # surface area of plastic particle [m2]
    r_a = ((3./4.)*(v_a/math.pi))**(1./3.)               # radius of an algal cell [m]

    v_bfa = (v_a*a)*theta_pl                              # volume of living biofilm [m3]

    v_cy = (4./3.)*math.pi*(r_a*50./60.)**3               # volume of cytoplasm [m3] of a single algal cell ~59/60 Miklasz & Denny 2010
    v_fr = v_a-v_cy                                       # volume of the frustule [m3] of a single algal cell ~1/60

    v_bfd = (v_fr*a_dead)*theta_pl                         # volume of dead biofilm [m3]
    v_tot = v_bfa + v_bfd + v_pl                           # volume of total [m3]
    t_bf = ((v_tot*(3./(4.*math.pi)))**(1./3.))-particle.r_pl  # biofilm thickness [m]

    #------ Diffusivity -----
    r_tot = particle.r_pl + t_bf                              # total radius [m]
    rho_tot = (v_pl * particle.rho_pl + v_bfa*rho_bf + v_bfd*rho_fr)/v_tot # total density [kg m-3] dead cells = frustule
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
    a_growth = mu_aa*a                                        # [no. m-2 s-1]

    a_grazing = gr_ad*a                                       # grazing losses [no. m-2 s-1]
    a_linear = fieldset.mu1*a                                 # linear losses [no. m-2 s-1] eq 67 Yool et al. 2013
    a_non_linear = a_nlin*a                                   # non-linear losses [no. m-2 s-1] eq 72 Yool et al. 2013
    a_resp = (q10**((t-20.)/10.))*r20*a                       # [no. m-2 s-1] respiration

    particle.a_coll = a_coll
    particle.a_growth = a_growth
    particle.a_gr = a_grazing
    particle.a_l = a_linear
    particle.a_nl = a_non_linear
    particle.a_resp = a_resp
    particle.a += (a_coll + a_growth - a_grazing - a_resp - a_non_linear) * particle.dt

    a_diss = a_diss*a_dead                     # [no. m-2 s-1]
    a_indirect = fieldset.D3*a_grazing         # [no. m-2 s-1]
    a_direct = fieldset.D1*a_non_linear        # [no. m-2 s-1]
    particle.a_direct = a_direct
    particle.a_indirect = a_indirect
    particle.a_diss = a_diss

    particle.a_dead += (a_direct + a_indirect - a_diss) * particle.dt # From eq. 90 Yool et al. 2013

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
    if z0 <=0.6: # NEMO's 'surface depth'
        vs = 0
        particle.depth = 0.6
    elif z0 > 0:
        particle.depth += vs * particle.dt

    particle.vs = vs
    particle.rho_tot = rho_tot
    particle.r_tot = r_tot
    particle.delta_rho = delta_rho
    particle.t_bf = t_bf

def AdvectionRK4_3D_vert(particle, fieldset, time):
    """Advection of particles using fourth-order Runge-Kutta integration including vertical velocity.
    Function needs to be converted to Kernel object before execution"""
    (w1) = fieldset.W[time, particle.depth, particle.lat, particle.lon]
    #lon1 = particle.lon + u1*.5*particle.dt
    #lat1 = particle.lat + v1*.5*particle.dt
    dep1 = particle.depth + w1*.5*particle.dt
    (w2) = fieldset.W[time + .5 * particle.dt, dep1, particle.lat, particle.lon]
    #lon2 = particle.lon + u2*.5*particle.dt
    #lat2 = particle.lat + v2*.5*particle.dt
    dep2 = particle.depth + w2*.5*particle.dt
    (w3) = fieldset.W[time + .5 * particle.dt, dep2, particle.lat, particle.lon]
    #lon3 = particle.lon + u3*particle.dt
    #lat3 = particle.lat + v3*particle.dt
    dep3 = particle.depth + w3*particle.dt
    (w4) = fieldset.W[time + particle.dt, dep3, particle.lat, particle.lon]
    #particle.lon += particle.lon #(u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
    #particle.lat += particle.lat #lats[1,1] #(v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt
    particle.depth += (w1 + 2*w2 + 2*w3 + w4) / 6. * particle.dt

def DeleteParticle(particle, fieldset, time):
    """Kernel for deleting particles if they are out of bounds."""
    print('particle is deleted out of bounds at lon = '+str(particle.lon)+', lat ='+str(particle.lat)+', depth ='+str(particle.depth))
    particle.delete() 
    
def DeleteParticleInterp(particle, fieldset, time):
    """Kernel for deleting particles if they are out of bounds."""
    print('particle is deleted due to an interpolation error at lon = '+str(particle.lon)+', lat ='+str(particle.lat)+', depth ='+str(particle.depth))
    particle.delete()

def getclosest_ij(lats,lons,latpt,lonpt):     
    """Function to find the index of the closest point to a certain lon/lat value."""
    dist_sq = (lats-latpt)**2 + (lons-lonpt)**2                 # find squared distance of every point on grid
    minindex_flattened = dist_sq.argmin()                       # 1D index of minimum dist_sq element
    return np.unravel_index(minindex_flattened, lats.shape)     # Get 2D index for latvals and lonvals arrays from 1D index

def periodicBC(particle, fieldset, time):
    if particle.lon <= -180.:
        particle.lon += 360.
    elif particle.lon >= 180.:
        particle.lon -= 360.

def Profiles_full_grazing(particle, fieldset, time):
    particle.temp = fieldset.cons_temperature[time, particle.depth,particle.lat,particle.lon]
    particle.d_phy= fieldset.d_phy[time, particle.depth,particle.lat,particle.lon]
    particle.nd_phy = fieldset.nd_phy[time, particle.depth,particle.lat,particle.lon]
    particle.mic_zoo = fieldset.mic_zoo[time, particle.depth,particle.lat,particle.lon]
    particle.mes_zoo = fieldset.mes_zoo[time, particle.depth,particle.lat,particle.lon]
    particle.detr = fieldset.detr[time, particle.depth,particle.lat,particle.lon]
    particle.tpp3 = fieldset.tpp3[time,particle.depth,particle.lat,particle.lon]
    particle.euphz = fieldset.euph_z[time, particle.depth, particle.lat, particle.lon]
    particle.d_si = fieldset.Di_Si[time, particle.depth, particle.lat, particle.lon]

    mu_w = 4.2844E-5 + (1/((0.157*(particle.temp + 64.993)**2)-91.296))
    A = 1.541 + 1.998E-2*particle.temp - 9.52E-5*particle.temp**2
    B = 7.974 - 7.561E-2*particle.temp + 4.724E-4*particle.temp**2
    S_sw = fieldset.abs_salinity[time, particle.depth, particle.lat, particle.lon]/1000
    particle.sw_visc = mu_w*(1 + A*S_sw + B*S_sw**2)
    particle.kin_visc = particle.sw_visc/particle.density
    particle.w_adv = fieldset.W[time,particle.depth,particle.lat,particle.lon]

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

def markov_0_KPP_reflect(particle, fieldset, time):
    """
    If a particle tries to cross the boundary, then it is reflected back
    If a particle tries to cross the boundary, then it is reflected back
    Author: Victor Onink
    Adapted 1D -> 3D
    """
    g= fieldset.G
    rho_a = fieldset.Rho_a
    wave_age = fieldset.Wave_age
    phi = fieldset.Phi
    vk = fieldset.Vk

    rho_sw = particle.density
    mld = fieldset.mldr[time, particle.depth, particle.lat, particle.lon]
    particle.tau = fieldset.taum[time, particle.depth, particle.lat, particle.lon]
    particle.mld = particle.depth/mld
    particle.w10 = fieldset.w_10[time, particle.depth, particle.lat, particle.lon]

    # Define KPP profile from tau and mld
    u_s_a =  math.sqrt(particle.tau/rho_a)
    u_s_w =  math.sqrt(particle.tau/rho_sw)
   
    alpha_dt = (vk * u_s_w) / (phi * mld ** 2)
    alpha = (vk * u_s_w) / phi

    beta = wave_age * u_s_a / particle.w10
    z0 = 3.5153e-5 * math.pow(beta, -0.42) * math.pow(particle.w10, 2) / g 
        
    if particle.mld<1:
        dK_z_p = alpha_dt * (mld - particle.depth) * (mld -3 * particle.depth -2 * z0)
    else:
        dK_z_p = 0

    particle.KPP = alpha * (particle.depth + 0.5 * dK_z_p * particle.dt +z0) * math.pow(1 - (particle.depth + 0.5 * dK_z_p * particle.dt)/ mld, 2)
    if particle.mld<1:
        K_z = particle.KPP
    else:
        K_z = 0.
    
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
    if potential < 0.6:
        particle.depth = 0.6 + (0.6 - potential)
    elif potential > 0:
        particle.depth = potential

def markov_0_KPP_ceiling_tides(particle, fieldset, time):
    """
    If a particle tries to cross the boundary, then it stays at the surface
    Author: Victor Onink
    Adapted 1D -> 3D
    """
    g= fieldset.G
    rho_a = fieldset.Rho_a
    wave_age = fieldset.Wave_age
    phi = fieldset.Phi
    vk = fieldset.Vk

    rho_sw = particle.density
    mld = fieldset.mldr[time, particle.depth, particle.lat, particle.lon]
    particle.tau = fieldset.taum[time, particle.depth, particle.lat, particle.lon]
    particle.mld = particle.depth/mld
    particle.w10 = fieldset.w_10[time, particle.depth, particle.lat, particle.lon]

    # Define KPP profile from tau and mld
    u_s_a =  math.sqrt(particle.tau/rho_a)
    u_s_w =  math.sqrt(particle.tau/rho_sw)

    alpha_dt = (vk * u_s_w) / (phi * mld ** 2)
    alpha = (vk * u_s_w) / phi

    beta = wave_age * u_s_a / particle.w10
    z0 = 3.5153e-5 * math.pow(beta, -0.42) * math.pow(particle.w10, 2) / g

    if particle.mld<1:
        dK_z = alpha_dt * (mld - particle.depth) * (mld -3 * particle.depth -2 * z0)
    else:
        dK_z = 0

    particle.KPP = alpha * (particle.depth + 0.5 * dK_z * particle.dt +z0) * math.pow(1 - (particle.depth + 0.5 * dK_z * particle.dt)/ mld, 2)
    if particle.mld<1:
        K_z = particle.KPP
    else:
        K_z = 0.

    particle.dK_z_t = fieldset.dKzdz[time, particle.depth, particle.lat, particle.lon]
    particle.K_z_t = fieldset.Kz[time, particle.depth, particle.lat, particle.lon]

    dK_z_t = particle.dK_z_t
    K_z_t = particle.K_z_t

    K_z += K_z_t
    dK_z += dK_z_t

    # According to Ross & Sharples (2004), first the deterministic part of equation 1
    deterministic = dK_z * particle.dt

    # The random walk component
    R = ParcelsRandom.uniform(-1., 1.) * math.sqrt(math.fabs(particle.dt) * 3)
    bz = math.sqrt(2 * K_z)

    # Total movement
    w_m_step = deterministic + R * bz
    particle.w_m = w_m_step/particle.dt

    # The ocean surface acts as a lid off of which the plastic bounces if tries to cross the ocean surface
    potential = particle.depth + w_m_step
    if potential < 0.6:
        particle.depth = 0.6
    elif potential > 0:
        particle.depth = potential

def tidal_diffusivity(particle, fieldset, time):
    """
    If a particle tries to cross the boundary, then it is reflected back
    Author: Victor Onink
    Adapted 1D -> 3D
    """
    particle.dK_z_t = fieldset.dKzdz[time, particle.depth, particle.lat, particle.lon]
    particle.K_z_t = fieldset.Kz[time, particle.depth, particle.lat, particle.lon]

    dK_z = particle.dK_z_t
    K_z = particle.K_z_t

    # According to Ross & Sharples (2004), first the deterministic part of equation 1
    deterministic = dK_z * particle.dt

    # The random walk component
    R = ParcelsRandom.uniform(-1., 1.) * math.sqrt(math.fabs(particle.dt) * 3)
    bz = math.sqrt(2 * K_z)

    # Total movement
    w_t_step = deterministic + R * bz
    particle.w_m_b = w_t_step/particle.dt
    # The ocean surface acts as a lid off of which the plastic bounces if tries to cross the ocean surface
    potential = particle.depth + w_t_step
    if potential < 0.6:
        particle.depth = 0.6 + (0.6 - potential)
    elif potential > 0.6 and potential<5000:
        particle.depth = potential

""" Defining the particle class """

class plastic_particle(JITParticle): #ScipyParticle): #
    u = Variable('u', dtype=np.float32,to_write=False)
    v = Variable('v', dtype=np.float32,to_write=False)
    w = Variable('w', dtype=np.float32,to_write=False)
    w_adv = Variable('w_adv', dtype=np.float32,to_write=True)
    temp = Variable('temp',dtype=np.float32,to_write=False)
    density = Variable('density',dtype=np.float32,to_write=False)
    tpp3 = Variable('tpp3',dtype=np.float32,to_write=True)
    d_phy = Variable('d_phy',dtype=np.float32,to_write=True)
    nd_phy = Variable('nd_phy',dtype=np.float32,to_write=False)
    mic_zoo = Variable('mic_zoo',dtype=np.float32,to_write=False)
    mes_zoo = Variable('mes_zoo',dtype=np.float32,to_write=False)
    detr = Variable('detr',dtype=np.float32,to_write=False)
    a = Variable('a',dtype=np.float32,to_write=True)
    a_dead = Variable('a_dead', dtype=np.float32,to_write=True)
    a_coll = Variable('a_coll', dtype=np.float32, to_write=True)
    a_growth = Variable('a_growth', dtype=np.float32, to_write=True)
    a_resp = Variable('a_resp', dtype=np.float32, to_write=True)
    a_gr = Variable('a_gr', dtype=np.float32, to_write=True)
    a_l = Variable('a_l', dtype=np.float32, to_write=True)
    a_nl = Variable('a_nl', dtype=np.float32, to_write=True)
    a_direct = Variable('a_direct', dtype=np.float32, to_write=True)
    a_indirect = Variable('a_indirect', dtype=np.float32, to_write=True)
    a_diss = Variable('a_diss', dtype=np.float32, to_write=True)
    kin_visc = Variable('kin_visc',dtype=np.float32,to_write=False)
    sw_visc = Variable('sw_visc',dtype=np.float32,to_write=False)
    vs = Variable('vs',dtype=np.float32,to_write=True)
    w_m = Variable('w_m', dtype=np.float32, to_write=True)
    w_m_b = Variable('w_m_b', dtype=np.float32, to_write=True)
    mld = Variable('mld', dtype=np.float32, to_write=True)
    euphz = Variable('euphz', dtype=np.float32, to_write=True) 
    rho_tot = Variable('rho_tot',dtype=np.float32,to_write=True) 
    r_tot = Variable('r_tot',dtype=np.float32,to_write=True)
    delta_rho = Variable('delta_rho',dtype=np.float32,to_write=True)
    rho_bf = Variable('rho_bf',dtype=np.float32,to_write='once')
    t_bf = Variable('t_bf',dtype=np.float32,to_write=True)
    vs_init = Variable('vs_init',dtype=np.float32,to_write=True)
    KPP = Variable('KPP', dtype=np.float32, to_write=False)
    K_z_t = Variable('K_z_t', dtype=np.float32, to_write=False)
    dK_z_t = Variable('dK_z_t', dtype=np.float32, to_write=False)
    tau = Variable('tau', dtype=np.float32, to_write=False)
    w10 = Variable('w10', dtype=np.float32, to_write=False)
    r_pl = Variable('r_pl',dtype=np.float32,to_write='once')   
    rho_pl = Variable('rho_pl',dtype=np.float32,to_write='once')
    Si_N = Variable('Si_N',dtype=np.float32,to_write=False)
    d_si = Variable('d_si',dtype=np.float32,to_write=False)
    dstar = Variable('dstar', dtype=np.float32, to_write=True)
    wstar = Variable('wstar', dtype=np.float32, to_write=True)

if __name__ == "__main__":     
    p = ArgumentParser(description="""choose starting month and year""")
    p.add_argument('-mon', choices = ('01','12','03','06','09','10'), action="store", dest="mon", 
                   help='start month for the run')
    p.add_argument('-yr', choices = ('2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010'), action="store", dest="yr",
                   help='start year for the run')
    p.add_argument('-region', choices = ('NPSG','EqPac','SO'), action = "store", dest = "region", help ='region where particles released')
    p.add_argument('-mixing', choices = ('no', 'fixed', 'markov_0_KPP_reflect', 'markov_0_KPP_ceiling_tides'), action = "store", dest = 'mixing', help='Type of random vertical mixing. "no" is none, "fixed" is mld between 0.2 and -0.2 m/s')
    p.add_argument('-biofouling', choices=('MEDUSA', 'MEDUSA_detritus'), action='store', dest = 'biofouling')
    p.add_argument('-system', choices=('gemini', 'cartesius'), action='store', dest = 'system', help='"gemini" or "cartesius"')
    p.add_argument('-bg_mixing', choices=('no', 'tidal'), action='store', dest = 'bg_mixing')
    p.add_argument('-no_biofouling', choices =('True','False'), action="store", dest="no_biofouling", help='True if using Kooi kernel without biofouling')   
    p.add_argument('-no_advection', choices =('True','False'), action="store", dest="no_advection", help='True if removing advection_RK43D kernel')
 
    args = p.parse_args()
    mon = args.mon
    yr = args.yr
    region = args.region
    mixing = args.mixing
    biofouling = args.biofouling
    no_biofouling = False #no_biofouling = args.no_biofouling
    no_advection = str(args.no_advection)
    system = args.system
 
    """ Load particle release locations from plot_NEMO_landmask.ipynb """
    # CHOOSE

    #------ Fieldset grid  ------
    if region == 'NPSG' and no_advection == 'False':
        minlat = -10 
        maxlat = 70 
        minlon = 110 # -180 #75 
        maxlon = -100 #45   
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
        maxlat = -25
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
    else:
        print('Error: no valid system argument parsed')  

    if mon =='12' and simdays>365:
        yr0 = str(int(yr)-1)
        yr2 = str(int(yr)+1)
        ufiles = (sorted(glob(dirread+'ORCA0083-N06_'+yr0+'1*d05U.nc'))+ sorted(glob(dirread+'ORCA0083-N06_'+yr+'*d05U.nc'))+ sorted(glob(dirread+'ORCA0083-N06_'+yr2+'*d05U.nc')))
        vfiles = (sorted(glob(dirread+'ORCA0083-N06_'+yr0+'1*d05V.nc'))+ sorted(glob(dirread+'ORCA0083-N06_'+yr+'*d05V.nc'))+ sorted(glob(dirread+'ORCA0083-N06_'+yr2+'*d05V.nc')))
        wfiles = (sorted(glob(dirread+'ORCA0083-N06_'+yr0+'1*d05W.nc'))+ sorted(glob(dirread+'ORCA0083-N06_'+yr+'*d05W.nc'))+ sorted(glob(dirread+'ORCA0083-N06_'+yr2+'*d05W.nc')))
        pfiles = (sorted(glob(dirread_bgc+'ORCA0083-N06_'+yr0+'1*d05P.nc'))+ sorted(glob(dirread_bgc+'ORCA0083-N06_'+yr+'*d05P.nc'))+ sorted(glob(dirread_bgc+'ORCA0083-N06_'+yr2+'*d05P.nc')))
        ppfiles = (sorted(glob(dirread_bgc+'ORCA0083-N06_'+yr0+'1*d05D.nc'))+ sorted(glob(dirread_bgc+'ORCA0083-N06_'+yr+'*d05D.nc'))+ sorted(glob(dirread_bgc+'ORCA0083-N06_'+yr2+'*d05D.nc')))
        tsfiles = (sorted(glob(dirread+'ORCA0083-N06_'+yr0+'1*d05T.nc'))+ sorted(glob(dirread+'ORCA0083-N06_'+yr+'*d05T.nc'))+ sorted(glob(dirread+'ORCA0083-N06_'+yr2+'*d05T.nc')))
    elif mon =='12':
        yr0 = str(int(yr)-1)
        ufiles = (sorted(glob(dirread+'ORCA0083-N06_'+yr0+'1*d05U.nc'))+ sorted(glob(dirread+'ORCA0083-N06_'+yr+'*d05U.nc')))
        vfiles = (sorted(glob(dirread+'ORCA0083-N06_'+yr0+'1*d05V.nc'))+ sorted(glob(dirread+'ORCA0083-N06_'+yr+'*d05V.nc')))
        wfiles = (sorted(glob(dirread+'ORCA0083-N06_'+yr0+'1*d05W.nc'))+ sorted(glob(dirread+'ORCA0083-N06_'+yr+'*d05W.nc')))
        pfiles = (sorted(glob(dirread_bgc+'ORCA0083-N06_'+yr0+'1*d05P.nc'))+ sorted(glob(dirread_bgc+'ORCA0083-N06_'+yr+'*d05P.nc')))
        ppfiles = (sorted(glob(dirread_bgc+'ORCA0083-N06_'+yr0+'1*d05D.nc'))+ sorted(glob(dirread_bgc+'ORCA0083-N06_'+yr+'*d05D.nc')))
        tsfiles = (sorted(glob(dirread+'ORCA0083-N06_'+yr0+'1*d05T.nc'))+ sorted(glob(dirread+'ORCA0083-N06_'+yr+'*d05T.nc')))
    elif mon == '10' and simdays>450:
        yr1 = str(int(yr)+1)
        yr2 = str(int(yr)+2)
        yr0 = yr
        ufiles = (sorted(glob(dirread+'ORCA0083-N06_'+yr+'*d05U.nc'))+ sorted(glob(dirread+'ORCA0083-N06_'+yr1+'0*d05U.nc'))+ sorted(glob(dirread+'ORCA0083-N06_'+yr2+'0*d05U.nc')))
        vfiles = (sorted(glob(dirread+'ORCA0083-N06_'+yr+'*d05V.nc'))+ sorted(glob(dirread+'ORCA0083-N06_'+yr1+'0*d05V.nc'))+ sorted(glob(dirread+'ORCA0083-N06_'+yr2+'0*d05V.nc')))
        wfiles = (sorted(glob(dirread+'ORCA0083-N06_'+yr+'*d05W.nc'))+ sorted(glob(dirread+'ORCA0083-N06_'+yr1+'0*d05W.nc'))+ sorted(glob(dirread+'ORCA0083-N06_'+yr2+'0*d05W.nc')))
        pfiles = (sorted(glob(dirread_bgc+'ORCA0083-N06_'+yr+'*d05P.nc'))+ sorted(glob(dirread_bgc+'ORCA0083-N06_'+yr1+'0*d05P.nc'))+ sorted(glob(dirread_bgc+'ORCA0083-N06_'+yr2+'0*d05P.nc')))
        ppfiles = (sorted(glob(dirread_bgc+'ORCA0083-N06_'+yr+'*d05D.nc'))+ sorted(glob(dirread_bgc+'ORCA0083-N06_'+yr1+'0*d05D.nc'))+ sorted(glob(dirread_bgc+'ORCA0083-N06_'+yr2+'0*d05D.nc')))
        tsfiles = (sorted(glob(dirread+'ORCA0083-N06_'+yr+'*d05T.nc'))+ sorted(glob(dirread+'ORCA0083-N06_'+yr1+'0*d05T.nc'))+ sorted(glob(dirread+'ORCA0083-N06_'+yr2+'0*d05T.nc')))
    elif mon == '09':
        yr1 = str(int(yr)+1)
        yr0 = yr
        ufiles = (sorted(glob(dirread+'ORCA0083-N06_'+yr+'*d05U.nc'))+ sorted(glob(dirread+'ORCA0083-N06_'+yr1+'0*d05U.nc')))
        vfiles = (sorted(glob(dirread+'ORCA0083-N06_'+yr+'*d05V.nc'))+ sorted(glob(dirread+'ORCA0083-N06_'+yr1+'0*d05V.nc')))
        wfiles = (sorted(glob(dirread+'ORCA0083-N06_'+yr+'*d05W.nc'))+ sorted(glob(dirread+'ORCA0083-N06_'+yr1+'0*d05W.nc')))
        pfiles = (sorted(glob(dirread_bgc+'ORCA0083-N06_'+yr+'*d05P.nc'))+ sorted(glob(dirread_bgc+'ORCA0083-N06_'+yr1+'0*d05P.nc')))
        ppfiles = (sorted(glob(dirread_bgc+'ORCA0083-N06_'+yr+'*d05D.nc'))+ sorted(glob(dirread_bgc+'ORCA0083-N06_'+yr1+'0*d05D.nc')))
        tsfiles = (sorted(glob(dirread+'ORCA0083-N06_'+yr+'*d05T.nc'))+ sorted(glob(dirread+'ORCA0083-N06_'+yr1+'0*d05T.nc')))
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
    
    initialgrid_mask = dirread+'ORCA0083-N06_20070105d05U.nc'
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
    
    fieldset = FieldSet.from_nemo(filenames, variables, dimensions, allow_time_extrapolation=False, chunksize=chs, indices = indices)

    variable = ('Kz', 'TIDAL_Kz')
    dimension = {'lon': 'Longitude', 'lat': 'Latitude', 'depth':'Depth_midpoint'}
    if system == 'gemini':
        Kz_field = Field.from_netcdf('/scratch/rfischer/Kooi_data/data_input/Kz.nc', variable, dimension)
    elif system == 'cartesius':
        Kz_field = Field.from_netcdf('~/biofouling_3dtransport_2/Preprocessing/Kz.nc', variable, dimension)
    fieldset.add_field(Kz_field)
    variabled = ('dKzdz', 'TIDAL_dKz')
    if system == 'gemini':
        dKz_field = Field.from_netcdf('/scratch/rfischer/Kooi_data/data_input/Kz.nc', variabled, dimension)
    elif system == 'cartesius':
        dKz_field = Field.from_netcdf('~/biofouling_3dtransport_2/Preprocessing/Kz.nc', variabled, dimension)
    fieldset.add_field(dKz_field)

    # ------ Defining constants ------
    fieldset.add_constant('Gr_a', 0.39 / 86400.)
    fieldset.add_constant('collision_eff', 1.)
    fieldset.add_constant('K', 1.0306E-13 / (86400. ** 2.))  # Boltzmann constant [m2 kg d-2 K-1] now [s-2] (=1.3804E-23)
    fieldset.add_constant('Rho_bf', 1388.)                   # density of biofilm [g m-3]
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

    if mixing == 'markov_0_KPP_reflect' or mixing == 'markov_0_KPP_ceiling_tides':
        fieldset.add_constant('Vk', 0.4)
        fieldset.add_constant('Phi', 0.9)
        fieldset.add_constant('Rho_a', 1.22)
        fieldset.add_constant('Wave_age', 35)

    lons = fieldset.U.lon
    lats = fieldset.U.lat
    depths = fieldset.U.depth

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

    rho_pls = np.tile(920, [n_res, n_res, n_sizebins*n_particles_per_bin])
    r_pls = uniform_release(n_locs, n_particles_per_bin, n_sizebins)

    pset = ParticleSet.from_list(fieldset=fieldset,         # the fields on which the particles are advected
                                 pclass=plastic_particle,   # the type of particles (JITParticle or ScipyParticle)
                                 lon= lon_release, #-160.,  # a vector of release longitudes 
                                 lat= lat_release, #36., 
                                 time = np.datetime64('%s-%s-01' % (yr0, mon)),
                                 depth = z_release,
                                 r_pl = r_pls,
                                 rho_pl = rho_pls,
                                 r_tot = r_pls,
                                 rho_tot = rho_pls)


    """ Kernal + Execution"""
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
    
    if no_advection == 'True':
        proc = 'bfnoadv'
        kernels = pset.Kernel(PolyTEOS10_bsq)+ pset.Kernel(Profiles_full_grazing) +pset.Kernel(AdvectionRK4_3D_vert) + pset.Kernel(periodicBC)
    elif no_advection == 'False':
        proc = 'bfadv'
        kernels = pset.Kernel(PolyTEOS10_bsq)+ pset.Kernel(Profiles_full_grazing) + pset.Kernel(AdvectionRK4_3D) + pset.Kernel(periodicBC) 
    else:
        print(no_advection+' is not a correct argument')
        
    if mixing == 'fixed':
        kernels += pset.Kernel(vertical_mixing_random_constant)
    elif mixing == 'markov_0_KPP_reflect':
        kernels += pset.Kernel(markov_0_KPP_reflect)
    elif mixing == 'markov_0_KPP_ceiling_tides':
        kernels += pset.Kernel(markov_0_KPP_ceiling_tides)

    if biofouling == 'MEDUSA':
        kernels += pset.Kernel(MEDUSA_full_grazing)
    elif biofouling == 'MEDUSA_detritus':
        kernels += pset.Kernel(MEDUSA_detritus_full_grazing)

    if system == 'cartesius':
        outfile = '/scratch-local/rfischer/regional_'+region+'_'+proc+'_'+s+'_'+yr+'_'+res+'res_'+mixing+'_'+biofouling+'_'+str(int(fieldset.Rho_bf))+'rhobf_'+str(round(simdays,2))+'days_'+str(secsdt)+'dtsecs_'+str(round(hrsoutdt,2))+'hrsoutdt' 
    elif system == 'gemini':
        outfile = '/scratch/rfischer/Kooi_data/data_output/regional_'+region+'_'+proc+'_'+s+'_'+yr+'_'+res+'res_'+mixing+'NEMO_'+str(int(fieldset.Rho_bf))+'rhobf_'+str(round(simdays,2))+'days_'+str(secsdt)+'dtsecs_'+str(round(hrsoutdt,2))+'hrsoutdt'


    pfile= ParticleFile(outfile, pset, outputdt=delta(hours = hrsoutdt))
    pfile.add_metadata('collision efficiency', str(1.))

    pset.execute(kernels, runtime=delta(days=simdays), dt=delta(seconds = secsdt), output_file=pfile, verbose_progress=True, recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle, ErrorCode.ErrorInterpolation: DeleteParticleInterp})

    pfile.close()

    print('Execution finished')
