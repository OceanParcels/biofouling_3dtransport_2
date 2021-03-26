"""
Authors: Delphine Lobelle, Reint Fischer

Executable python script to simulate regional biofouling particles with parameterized wind and tidal mixing.
"""

from parcels import FieldSet, ParticleSet, JITParticle, ScipyParticle, AdvectionRK4_3D, AdvectionRK4, ErrorCode, ParticleFile, Variable, Field, NestedField, VectorField, timer, ParcelsRandom 
from parcels.kernels.TEOSseawaterdensity import PolyTEOS10_bsq
from datetime import timedelta as delta
import numpy as np
from numpy.random import default_rng
from glob import glob
import xarray as xr
import warnings
from numpy import *
import math as math
from argparse import ArgumentParser
warnings.filterwarnings("ignore")

seed = 123
ParcelsRandom.seed(seed)
rng = default_rng(seed)

#------ Choose ------:
simdays = 6
secsdt = 60 #30
hrsoutdt = 1

"""functions and kernels"""

def Kooi(particle,fieldset,time):  
    """
    Kernel to compute the vertical velocity (Vs) of particles due to changes in ambient algal concentrations, growth and death of attached algae based on Kooi et al. 2017 model 
    """
    # ------ Constants and algal properties -----
    g = fieldset.G            # gravitational acceleration [m s-2]
    k = fieldset.K            # Boltzmann constant [m2 kg d-2 K-1] now [s-2] (=1.3804E-23)
    rho_bf = fieldset.Rho_bf  # density of biofilm [g m-3]
    v_a = fieldset.V_a        # Volume of 1 algal cell [m-3]
    gr_a = fieldset.Gr_a      # grazing rate [s-1]
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
    mu_n2 = mu_n/aa                   # conversion from [no. m-3 d-1] to [d-1]
    
    if mu_n2<0.:
        mu_aa = 0.
    elif mu_n2>1.85:
        mu_aa = 1.85/86400.           # maximum growth rate
    else:
        mu_aa = mu_n2/86400.          # conversion from d-1 to s-1
    
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
    a_coll = (beta_a*ad)/theta_pl*fieldset.collision_eff      # [no. m-2 s-1] collisions with diatoms
    a_growth = mu_aa*a
    a_grazing = gr_a*a
    a_resp = (q10**((t-20.)/10.))*r20*a
    
    particle.a_coll = a_coll
    particle.a_growth = a_growth
    particle.a_resp = a_resp
    particle.a += (a_coll + a_growth - a_grazing - a_resp) * particle.dt

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

def Kooi_suddendeath(particle,fieldset,time):
    """
    Kernel to compute the vertical velocity (Vs) of particles due to changes in ambient algal concentrations, growth and death of attached algae based on Kooi et al. 2017 model
    """
    # ------ Constants and algal properties -----
    g = fieldset.G            # gravitational acceleration [m s-2]
    k = fieldset.K            # Boltzmann constant [m2 kg d-2 K-1] now [s-2] (=1.3804E-23)
    if particle.depth > particle.euphz:
        rho_bf = fieldset.Rho_fr  # frustule density [g m-3]
    else:
        rho_bf = fieldset.Rho_bf  # density of biofilm [g m-3]
    v_a = fieldset.V_a        # Volume of 1 algal cell [m-3]
    gr_a = fieldset.Gr_a      # grazing rate [s-1]
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
    mu_n2 = mu_n/aa                   # conversion from [no. m-3 d-1] to [d-1]

    if mu_n2<0.:
        mu_aa = 0.
    elif mu_n2>1.85:
        mu_aa = 1.85/86400.           # maximum growth rate
    else:
        mu_aa = mu_n2/86400.          # conversion from d-1 to s-1

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
    a_coll = (beta_a*ad)/theta_pl*fieldset.collision_eff      # [no. m-2 s-1] collisions with diatoms
    a_growth = mu_aa*a
    a_grazing = gr_a*a
    a_resp = (q10**((t-20.)/10.))*r20*a

    particle.a_coll = a_coll
    particle.a_growth = a_growth
    particle.a_resp = a_resp
    particle.a += (a_coll + a_growth - a_grazing - a_resp) * particle.dt

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
    
def Kooi_stress(particle,fieldset,time):
    """
    Kernel to compute the vertical velocity (Vs) of particles due to changes in ambient algal concentrations, growth and death of attached algae based on Kooi et al. 2017 model
    """
    # ------ Constants and algal properties -----
    g = fieldset.G            # gravitational acceleration [m s-2]
    k = fieldset.K            # Boltzmann constant [m2 kg d-2 K-1] now [s-2] (=1.3804E-23)
    rho_fr = fieldset.Rho_fr  # frustule density [g m-3] - now used as dead diatom density
    rho_bf = fieldset.Rho_bf  # density of living biofilm [g m-3]
    v_a = fieldset.V_a        # Volume of 1 algal cell [m-3]
    gr_a = fieldset.Gr_a      # grazing rate [s-1]
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
    a_dead = particle.a_dead # [no. m-2]
    vs = particle.vs  # [m s-1]

    #------ Nitrogen to cell ratios for ambient algal concentrations ('aa') and algal growth ('mu_aa') from NEMO output (no longer using N:C:AA (Redfield ratio), directly N:AA from Menden-Deuer and Lessard 2000)
    med_N2cell = 356.04e-09 # [mgN cell-1] median value is used below (as done in Kooi et al. 2017)
    wt_N = fieldset.Wt_N    # atomic weight of 1 mol of N = 14.007 g

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
    mu_n2 = mu_n/aa                   # conversion from [no. m-3 d-1] to [d-1]

    if mu_n2<0.:
        mu_aa = 0.
    elif mu_n2>1.85:
        mu_aa = 1.85/86400.           # maximum growth rate
    else:
        mu_aa = mu_n2/86400.          # conversion from d-1 to s-1

    #------ Volumes -----
    v_pl = (4./3.)*math.pi*particle.r_pl**3.             # volume of plastic [m3]
    theta_pl = 4.*math.pi*particle.r_pl**2.              # surface area of plastic particle [m2]
    r_a = ((3./4.)*(v_a/math.pi))**(1./3.)               # radius of algae [m]

    v_bfa = (v_a*a)*theta_pl                              # volume of living biofilm [m3]
    v_bfd = (v_a*a_dead)*theta_pl                         # volume of dead biofilm [m3]
    v_tot = v_bfa + v_bfd + v_pl                         # volume of total [m3]
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
    a_growth = mu_aa*a
    a_grazing = gr_a*a
    a_resp = (q10**((t-20.)/10.))*r20*a

    #----- Algal stress and death -----
    if particle.depth>particle.euphz:
        a_stress = 1.
        a_resp = 0.
        a_grazing = 0.
    else:
        a_stress = 0.
    a_death = a_stress*a

    particle.a_coll = a_coll
    particle.a_growth = a_growth
    particle.a_resp = a_resp
    particle.a += (a_coll + a_growth - a_grazing - a_resp - a_death) * particle.dt

    a_dead_grazing = gr_a*a_dead
    a_dead_resp = (q10**((t-20.)/10))*r20*a_dead
    particle.a_dead += (a_death - a_dead_grazing - a_dead_resp) * particle.dt

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
           
def Profiles(particle, fieldset, time):  
    particle.temp = fieldset.cons_temperature[time, particle.depth,particle.lat,particle.lon]  
    particle.d_phy= fieldset.d_phy[time, particle.depth,particle.lat,particle.lon]
    particle.nd_phy = fieldset.nd_phy[time, particle.depth,particle.lat,particle.lon]
    particle.tpp3 = fieldset.tpp3[time,particle.depth,particle.lat,particle.lon]
    particle.euphz = fieldset.euph_z[time, particle.depth, particle.lat, particle.lon]
 
    mu_w = 4.2844E-5 + (1/((0.157*(particle.temp + 64.993)**2)-91.296))
    A = 1.541 + 1.998E-2*particle.temp - 9.52E-5*particle.temp**2
    B = 7.974 - 7.561E-2*particle.temp + 4.724E-4*particle.temp**2
    S_sw = fieldset.abs_salinity[time, particle.depth, particle.lat, particle.lon]/1000
    particle.sw_visc = mu_w*(1 + A*S_sw + B*S_sw**2)
    particle.kin_visc = particle.sw_visc/particle.density
    particle.w_adv = fieldset.W[time,particle.depth,particle.lat,particle.lon]

def select_from_Cozar_random_continuous(n_particles_per_bin, bins, exponent):
    '''
    Create a set of particle radii by randomly drawing between given binedges from a power law distribution with given exponent.
    '''
    r_pls = np.zeros((len(bins)-1,n_particles_per_bin))
    for i in range(len(bins)-1):
        rnd = rng.random(n_particles_per_bin) #random number between 0 and 1
        rmin, rmax = bins[i]**exponent, bins[i+1]**exponent
        r_pls[i] = (rmin + (rmax - rmin)*rnd)**(1./exponent)
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


def vertical_mixing_random_constant(particle, fieldset, time):
    mld = fieldset.mldr[time, particle.depth, particle.lat, particle.lon]
    particle.mld = particle.depth/mld
    if particle.mld < 1:
        vmax = 0.02                                # [m/s] Maximum velocity
        particle.w_m = vmax*2*(ParcelsRandom.random()-0.5)  # [m/s] vertical mixing velocity
        z_0 = particle.depth + particle.w_m*particle.dt
        if z_0 <= 0.6:                              # [m] NEMO's surface depth
            particle.depth = 0.6
        else:
            particle.depth = z_0
    else:
        particle.w_m = 0

def mixed_layer(particle, fieldset, time):
    particle.mld = particle.depth/fieldset.mldr[time, particle.depth, particle.lat, particle.lon]

def markov_0_KPP_reflect(particle, fieldset, time):
    """
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
    u_s_w = math.sqrt(particle.tau/rho_sw)
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
    else:
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
    else:
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
    nd_phy = Variable('nd_phy',dtype=np.float32,to_write=True)
    a = Variable('a',dtype=np.float32,to_write=True)
    a_dead = Variable('a_dead', dtype=np.float32,to_write=True)
    a_coll = Variable('a_coll', dtype=np.float32, to_write=True)
    a_growth = Variable('a_growth', dtype=np.float32, to_write=True)
    a_resp = Variable('a_resp', dtype=np.float32, to_write=True)
    kin_visc = Variable('kin_visc',dtype=np.float32,to_write=False)
    sw_visc = Variable('sw_visc',dtype=np.float32,to_write=False)
    vs = Variable('vs',dtype=np.float32,to_write=True)
    w_m = Variable('w_m', dtype=np.float32, to_write=True)
    w_m_b = Variable('w_m_b', dtype=np.float32, to_write=True)
    mld = Variable('mld', dtype=np.float32, to_write=True)
    euphz = Variable('euphz', dtype=np.float32, to_write=True) 
    rho_tot = Variable('rho_tot',dtype=np.float32,to_write=False) 
    r_tot = Variable('r_tot',dtype=np.float32,to_write=True)
    delta_rho = Variable('delta_rho',dtype=np.float32,to_write=True)
    vs_init = Variable('vs_init',dtype=np.float32,to_write=True)
    KPP = Variable('KPP', dtype=np.float32, to_write=False)
    K_z_t = Variable('K_z_t', dtype=np.float32, to_write=False)
    dK_z_t = Variable('dK_z_t', dtype=np.float32, to_write=False)
    tau = Variable('tau', dtype=np.float32, to_write=False)
    w10 = Variable('w10', dtype=np.float32, to_write=False)
    r_pl = Variable('r_pl',dtype=np.float32,to_write='once')   
    rho_pl = Variable('rho_pl',dtype=np.float32,to_write='once')
    
if __name__ == "__main__":     
    p = ArgumentParser(description="""choose starting month and year""")
    p.add_argument('-mon', choices = ('01','12','03','06','09'), action="store", dest="mon", 
                   help='start month for the run')
    p.add_argument('-yr', choices = ('2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010'), action="store", dest="yr",
                   help='start year for the run')
    p.add_argument('-region', choices = ('NPSG','EqPac','SO'), action = "store", dest = "region", help ='region where particles released')
    p.add_argument('-a_grazing', choices = ('0.16', '0.39', '0.5'), action = "store", dest = 'grazing_rate', help='Grazing rate in d-1')
    p.add_argument('-mixing', choices = ('no', 'fixed', 'markov_0_KPP_reflect', 'markov_0_KPP_float'), action = "store", dest = 'mixing', help='Type of random vertical mixing. "no" is none, "fixed" is mld between 0.2 and -0.2 m/s')
    p.add_argument('-collision_eff', choices = ('1', '0.5'), default='1', action='store', dest='collision_eff', help='Collision efficiency: fraction of colliding algae that stick to the particle')
    p.add_argument('-system', choices=('gemini', 'cartesius'), action='store', dest = 'system', help='"gemini" or "cartesius"')
    p.add_argument('-bg_mixing', choices=('0', '0.00037', '0.00001', 'tidal'), action='store', dest = 'bg_mixing') 
    p.add_argument('-diatom_death', choices=('no', 'sudden', 'stress'), action='store', dest='diatom_death')
    p.add_argument('-no_biofouling', choices =('True','False'), action="store", dest="no_biofouling", help='True if using Kooi kernel without biofouling')   
    p.add_argument('-no_advection', choices =('True','False'), action="store", dest="no_advection", help='True if removing advection_RK43D kernel')
    
    args = p.parse_args()
    mon = args.mon
    yr = args.yr
    region = args.region
    grazing_rate = float(args.grazing_rate)
    mixing = args.mixing
    collision_eff = float(args.collision_eff)
    no_biofouling = False #no_biofouling = args.no_biofouling
    no_advection = str(args.no_advection)
    system = args.system
    if args.bg_mixing != 'tidal':
        bg_mixing = float(args.bg_mixing)
    else:
        bg_mixing = args.bg_mixing
    if args.diatom_death == 'no':
        diatom_death = ''
    else:
        diatom_death = args.diatom_death
 
    """ Load particle release locations from plot_NEMO_landmask.ipynb """
    # CHOOSE

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
#     elif mon == '01':
#         yr0 = yr
#         ufiles = sorted(glob(dirread+'ORCA0083-N06_'+yr+mon+'*d05U.nc'))
#         vfiles = sorted(glob(dirread+'ORCA0083-N06_'+yr+mon+'*d05V.nc'))
#         wfiles = sorted(glob(dirread+'ORCA0083-N06_'+yr+mon+'*d05W.nc'))
#         pfiles = sorted(glob(dirread_bgc+'ORCA0083-N06_'+yr+mon+'*d05P.nc'))
#         ppfiles = sorted(glob(dirread_bgc+'ORCA0083-N06_'+yr+mon+'*d05D.nc'))
#         tsfiles = sorted(glob(dirread+'ORCA0083-N06_'+yr+mon+'*d05T.nc'))
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
                 'euph_z': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': ppfiles}}

    variables = {'U': 'uo',
                 'V': 'vo',
                 'W': 'wo',
                 'd_phy': 'PHD',
                 'nd_phy': 'PHN',
                 'tpp3': 'TPP3', # units: mmolN/m3/d 
                 'cons_temperature': 'potemp',
                 'abs_salinity': 'salin',
                 'mldr': 'mldr10_1',
                 'taum': 'taum',
                 'w_10': 'sowindsp',
                 'euph_z': 'MED_XZE'}

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
                  'euph_z': {'lon':'glamf', 'lat':'gphif', 'time': 'time_counter'}}
    
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
           'nd_phy': {'time': ('time_counter', 1), 'depth': ('deptht', 25), 'lat': ('y', 200), 'lon': ('x', 200)},
           'tpp3': {'time': ('time_counter', 1), 'depth': ('deptht', 25), 'lat': ('y', 200), 'lon': ('x', 200)},
           'cons_temperature': {'time': ('time_counter', 1), 'depth': ('deptht', 25), 'lat': ('y', 200), 'lon': ('x', 200)},
           'abs_salinity': {'time': ('time_counter', 1), 'depth': ('deptht', 25), 'lat': ('y', 200), 'lon': ('x', 200)},
           'mldr': {'time': ('time_counter', 1), 'depth': ('deptht', 25), 'lat': ('y', 200), 'lon': ('x', 200)},
           'taum': {'time': ('time_counter', 1), 'depth': ('deptht', 25), 'lat': ('y', 200), 'lon': ('x', 200)},
           'w_10': {'time': ('time_counter', 1), 'depth': ('deptht', 25), 'lat': ('y', 200), 'lon': ('x', 200)},
           'euph_z': {'time': ('time_counter', 1), 'depth': ('deptht', 25), 'lat':('y', 200), 'lon': ('x', 200)}}
        
    fieldset = FieldSet.from_nemo(filenames, variables, dimensions, allow_time_extrapolation=False, chunksize=chs, indices = indices)

    if bg_mixing == 'tidal':
        variable = ('Kz', 'TIDAL_Kz')
        dimension = {'lon': 'Longitude', 'lat': 'Latitude', 'depth':'Depth_midpoint'}
        Kz_field = Field.from_netcdf('/scratch/rfischer/Kooi_data/data_input/Kz.nc', variable, dimension)
        fieldset.add_field(Kz_field)
        variabled = ('dKzdz', 'TIDAL_dKz')
        dKz_field = Field.from_netcdf('/scratch/rfischer/Kooi_data/data_input/Kz.nc', variabled, dimension)
        fieldset.add_field(dKz_field)

    # ------ Defining constants ------
    fieldset.add_constant('Gr_a', grazing_rate / 86400.)
    fieldset.add_constant('collision_eff', collision_eff)
    fieldset.add_constant('K', 1.0306E-13 / (86400. ** 2.))  # Boltzmann constant [m2 kg d-2 K-1] now [s-2] (=1.3804E-23)
    fieldset.add_constant('Rho_bf', 1388.)                   # density of biofilm [g m-3]
    fieldset.add_constant('Rho_fr', 2000.)                   # density of frustule [g m-3]
    fieldset.add_constant('Rho_cy', 1030.)                   # density of cytoplasm [g m-3] 
    fieldset.add_constant('V_a', 2.0E-16)                    # Volume of 1 algal cell [m-3]
    fieldset.add_constant('R20', 0.1 / 86400.)               # respiration rate, now [s-1]
    fieldset.add_constant('Q10', 2.)                         # temperature coefficient respiration [-]
    fieldset.add_constant('Gamma', 1.728E5 / 86400.)         # shear [d-1], now [s-1]
    fieldset.add_constant('Wt_N', 14.007)                    # atomic weight of nitrogen
    fieldset.add_constant('G', 7.32e10/(86400.**2.))
    
    if mixing == 'markov_0_KPP_reflect' or mixing == 'markov_0_KPP_float':
        fieldset.add_constant('Vk', 0.4)
        fieldset.add_constant('Phi', 0.9)
        if isinstance(bg_mixing, float) or isinstance(bg_mixing, int):
            fieldset.add_constant('Bulk_diff', bg_mixing)
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

    z_release = np.tile(0.6,[n_res,n_res, n_sizebins*n_particles_per_bin])
    res = '1x1'

    rho_pls = np.tile(920, [n_res, n_res, n_sizebins*n_particles_per_bin])
    r_pls = uniform_release(n_locs, n_particles_per_bin, n_sizebins)
    #r_pls = select_from_Cozar_random_continuous(lon_release.size,[5e-3, 5e-4, 5e-5, 5e-6, 5e-7],-3)

    pset = ParticleSet.from_list(fieldset=fieldset,         # the fields on which the particles are advected
                                 pclass=plastic_particle,   # the type of particles (JITParticle or ScipyParticle)
                                 lon= lon_release, #-160.,  # a vector of release longitudes 
                                 lat= lat_release, #36., 
                                 time = np.datetime64('%s-%s-05' % (yr0, mon)),
                                 depth = z_release,
                                 r_pl = r_pls,
                                 rho_pl = rho_pls,
                                 r_tot = r_pls,
                                 rho_tot = rho_pls)

    #for r_pl, rho_pl in zip(r_pls[1:], rho_pls[1:]):
    #    pset.add(ParticleSet.from_list(fieldset=fieldset,         # the fields on which the particles are advected
    #                             pclass=plastic_particle,   # the type of particles (JITParticle or ScipyParticle)
    #                             lon= lon_release, #-160.,  # a vector of release longitudes 
    #                             lat= lat_release, #36., 
    #                             time = np.datetime64('%s-%s-05' % (yr0, mon)),
    #                             depth = z_release,
    #                             r_pl = r_pl,
    #                             rho_pl = rho_pl * np.ones(np.array(lon_release).size),
    #                             r_tot = r_pl,
    #                             rho_tot = rho_pl * np.ones(np.array(lon_release).size)))


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
    
    if no_advection == 'True':
        proc = 'bfnoadv'
        kernels = pset.Kernel(AdvectionRK4_3D_vert) + pset.Kernel(periodicBC) +  pset.Kernel(PolyTEOS10_bsq)
    elif no_advection == 'False':
        proc = 'bfadv'
        kernels = pset.Kernel(AdvectionRK4_3D) + pset.Kernel(periodicBC) +  pset.Kernel(PolyTEOS10_bsq) 
    else:
        print(no_advection+' is not a correct argument')
        
    if mixing == 'fixed':
        kernels += pset.Kernel(vertical_mixing_random_constant)
    elif mixing == 'markov_0_KPP_reflect':
        kernels += pset.Kernel(markov_0_KPP_reflect)
    else:
        kernels += pset.Kernel(mixed_layer)
    if bg_mixing == 'tidal':
        kernels += pset.Kernel(tidal_diffusivity)
    kernels += pset.Kernel(Profiles)
    if diatom_death=='sudden':
        kernels += pset.Kernel(Kooi_suddendeath)
    elif diatom_death == 'stress':
        kernels += pset.Kernel(Kooi_stress)
    else:
        kernels += pset.Kernel(Kooi) 

    if system == 'cartesius':
        outfile = '/scratch-local/rfischer/Kooi_data/data_output/allrho/res_'+res+'/allr/regional_'+region+'_'+proc+'_'+s+'_'+yr+'_3D_grid'+res+'_allrho_allr_'+str(round(simdays,2))+'days_'+str(secsdt)+'dtsecs_'+str(round(hrsoutdt,2))+'hrsoutdt' 
    elif system == 'gemini':
        outfile = '/scratch/rfischer/Kooi_data/data_output/regional_'+region+'_'+proc+'_'+s+'_'+yr+'_'+res+'res_'+mixing+diatom_death+'_'+str(bg_mixing)+'mixing_'+str(round(simdays,2))+'days_'+str(secsdt)+'dtsecs_'+str(round(hrsoutdt,2))+'hrsoutdt'

    pfile= ParticleFile(outfile, pset, outputdt=delta(hours = hrsoutdt))
    pfile.add_metadata('collision efficiency', str(collision_eff))
    pfile.add_metadata('grazing rate', str(grazing_rate))
    pfile.add_metadata('background mixing', str(bg_mixing))

    pset.execute(kernels, runtime=delta(days=simdays), dt=delta(seconds = secsdt), output_file=pfile, verbose_progress=True, recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle, ErrorCode.ErrorInterpolation: DeleteParticleInterp})

    pfile.close()

    print('Execution finished')



