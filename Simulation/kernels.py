from parcels import JITParticle, Variable, ParcelsRandom
import numpy as np
import math

seed = 123
ParcelsRandom.seed(seed)

class plastic_particle(JITParticle):
    u = Variable('u', dtype=np.float32, to_write=False)
    v = Variable('v', dtype=np.float32, to_write=False)
    w = Variable('w', dtype=np.float32, to_write=False)
    w_adv = Variable('w_adv', dtype=np.float32, to_write=True)
    temp = Variable('temp', dtype=np.float32, to_write=False)
    density = Variable('density', dtype=np.float32, to_write=False)
    tpp3 = Variable('tpp3', dtype=np.float32, to_write=True)
    d_phy = Variable('d_phy', dtype=np.float32, to_write=True)
    nd_phy = Variable('nd_phy', dtype=np.float32, to_write=False)
    mic_zoo = Variable('mic_zoo', dtype=np.float32, to_write=False)
    mes_zoo = Variable('mes_zoo', dtype=np.float32, to_write=False)
    detr = Variable('detr', dtype=np.float32, to_write=False)
    a = Variable('a', dtype=np.float32, to_write=True)
    a_dead = Variable('a_dead', dtype=np.float32, to_write=True)
    a_coll = Variable('a_coll', dtype=np.float32, to_write=True)
    a_growth = Variable('a_growth', dtype=np.float32, to_write=True)
    a_resp = Variable('a_resp', dtype=np.float32, to_write=True)
    a_gr = Variable('a_gr', dtype=np.float32, to_write=True)
    a_l = Variable('a_l', dtype=np.float32, to_write=True)
    a_nl = Variable('a_nl', dtype=np.float32, to_write=True)
    a_direct = Variable('a_direct', dtype=np.float32, to_write=True)
    a_indirect = Variable('a_indirect', dtype=np.float32, to_write=True)
    a_diss = Variable('a_diss', dtype=np.float32, to_write=True)
    kin_visc = Variable('kin_visc', dtype=np.float32, to_write=False)
    sw_visc = Variable('sw_visc', dtype=np.float32, to_write=False)
    vs = Variable('vs', dtype=np.float32, to_write=True)
    w_m = Variable('w_m', dtype=np.float32, to_write=True)
    w_m_b = Variable('w_m_b', dtype=np.float32, to_write=True)
    mld = Variable('mld', dtype=np.float32, to_write=True)
    euphz = Variable('euphz', dtype=np.float32, to_write=True)
    rho_tot = Variable('rho_tot', dtype=np.float32, to_write=True)
    r_tot = Variable('r_tot', dtype=np.float32, to_write=True)
    delta_rho = Variable('delta_rho', dtype=np.float32, to_write=True)
    rho_bf = Variable('rho_bf', dtype=np.float32, to_write='once')
    t_bf = Variable('t_bf', dtype=np.float32, to_write=True)
    vs_init = Variable('vs_init', dtype=np.float32, to_write=True)
    KPP = Variable('KPP', dtype=np.float32, to_write=False)
    K_z_t = Variable('K_z_t', dtype=np.float32, to_write=False)
    dK_z_t = Variable('dK_z_t', dtype=np.float32, to_write=False)
    tau = Variable('tau', dtype=np.float32, to_write=False)
    w10 = Variable('w10', dtype=np.float32, to_write=False)
    r_pl = Variable('r_pl', dtype=np.float32, to_write='once')
    rho_pl = Variable('rho_pl', dtype=np.float32, to_write='once')
    Si_N = Variable('Si_N', dtype=np.float32, to_write=False)
    d_si = Variable('d_si', dtype=np.float32, to_write=False)
    dstar = Variable('dstar', dtype=np.float32, to_write=True)
    wstar = Variable('wstar', dtype=np.float32, to_write=True)

def profiles(particle, fieldset, time):
    particle.temp = fieldset.cons_temperature[time, particle.depth, particle.lat, particle.lon]
    particle.d_phy = fieldset.d_phy[time, particle.depth, particle.lat, particle.lon]
    particle.nd_phy = fieldset.nd_phy[time, particle.depth, particle.lat, particle.lon]
    particle.mic_zoo = fieldset.mic_zoo[time, particle.depth, particle.lat, particle.lon]
    particle.mes_zoo = fieldset.mes_zoo[time, particle.depth, particle.lat, particle.lon]
    particle.detr = fieldset.detr[time, particle.depth, particle.lat, particle.lon]
    particle.tpp3 = fieldset.tpp3[time, particle.depth, particle.lat, particle.lon]
    particle.euphz = fieldset.euph_z[time, particle.depth, particle.lat, particle.lon]
    particle.d_si = fieldset.Di_Si[time, particle.depth, particle.lat, particle.lon]

    mu_w = 4.2844E-5 + (1 / ((0.157 * (particle.temp + 64.993) ** 2) - 91.296))
    A = 1.541 + 1.998E-2 * particle.temp - 9.52E-5 * particle.temp ** 2
    B = 7.974 - 7.561E-2 * particle.temp + 4.724E-4 * particle.temp ** 2
    S_sw = fieldset.abs_salinity[time, particle.depth, particle.lat, particle.lon] / 1000
    particle.sw_visc = mu_w * (1 + A * S_sw + B * S_sw ** 2)
    particle.kin_visc = particle.sw_visc / particle.density
    particle.w_adv = fieldset.W[time, particle.depth, particle.lat, particle.lon]


def MEDUSA_biofouling(particle, fieldset, time):
    """
    Kernel to compute the vertical velocity (Vs) of particles due to changes in ambient algal concentrations, growth and death of attached algae based on Kooi et al. 2017 model settling velocity and MEDUSA 2.0 biofilm dynamics, including modelling of the 3D mesozooplankton grazing of diatoms
    """
    # ------ Constants and algal properties -----
    g = fieldset.G  # gravitational acceleration [m s-2]
    k = fieldset.K  # Boltzmann constant [m2 kg d-2 K-1] now [s-2] (=1.3804E-23)
    rho_fr = fieldset.Rho_fr  # frustule density [g m-3]
    rho_cy = fieldset.Rho_cy  # cytoplasm density [g m-3]
    v_a = fieldset.V_a  # Volume of 1 algal cell [m-3]
    r20 = fieldset.R20  # respiration rate [s-1]
    q10 = fieldset.Q10  # temperature coefficient respiration [-]
    gamma = fieldset.Gamma  # shear [s-1]

    # ------ Profiles from MEDUSA or Kooi theoretical profiles -----
    z = particle.depth  # [m]
    t = particle.temp  # [oC]
    sw_visc = particle.sw_visc  # [kg m-1 s-1]
    kin_visc = particle.kin_visc  # [m2 s-1]
    rho_sw = particle.density  # [kg m-3]
    a = particle.a  # [no. m-2]
    vs = particle.vs  # [m s-1]

    # ------ Nitrogen to cell ratios for ambient algal concentrations ('aa') and algal growth ('mu_aa') from NEMO output (no longer using N:C:AA (Redfield ratio), directly N:AA from Menden-Deuer and Lessard 2000)
    med_N2cell = 356.04e-09  # [mgN cell-1] median value is used below (as done in Kooi et al. 2017)
    wt_N = fieldset.Wt_N  # atomic weight of 1 mol of N = 14.007 g
    wt_Si = fieldset.Wt_Si  # atomic weight of 1 mor of Si = 28.0855

    # ------ Ambient algal concentration from MEDUSA's non-diatom + diatom phytoplankton
    n0 = particle.nd_phy + particle.d_phy  # [mmol N m-3] total plankton concentration engaging in primary production in MEDUSA
    d0 = particle.d_phy  # [mmol N m-3] diatom concentration that attaches to plastic particles

    n = n0 * wt_N  # conversion from [mmol N m-3] to [mg N m-3]
    d = d0 * wt_N  # conversion from [mmol N m-3] to [mg N m-3]

    n2 = n / med_N2cell  # conversion from [mg N m-3] to [no. m-3]
    d2 = d / med_N2cell  # conversion from [mg N m-3] to [no. m-3]

    if n2 < 0.:
        aa = 0.
    else:
        aa = n2  # [no m-3] ambient algae - to compare to Kooi model
    ad = d2  # [no m-3] ambient diatoms

    # ------ Primary productivity (algal growth) from MEDUSA TPP3
    tpp0 = particle.tpp3  # [mmol N m-3 d-1]
    mu_n0 = tpp0 * wt_N  # conversion from [mmol N m-3 d-1] to [mg N m-3 d-1] (atomic weight of 1 mol of N = 14.007 g)
    mu_n = mu_n0 / med_N2cell  # conversion from [mg N m-3 d-1] to [no. m-3 d-1]
    if aa > 0:  # If there are any ambient algae
        mu_n2 = mu_n / aa  # conversion from [no. m-3 d-1] to [d-1]
    else:
        mu_n2 = 0.

    if mu_n2 < 0.:
        mu_aa = 0.
    elif mu_n2 > 1.85:
        mu_aa = 1.85 / 86400.  # maximum growth rate
    else:
        mu_aa = mu_n2 / 86400.  # conversion from d-1 to s-1

    # ------ Grazing -----
    # Based on equations 54 and 55 in Yool et al. 2013
    FPn = fieldset.pmPn * math.pow(particle.nd_phy, 2)  # (mmol N m-3)**2 Interest in available non-diatoms
    FPd = fieldset.pmPd * math.pow(particle.d_phy, 2)  # (mmol N m-3)**2 Interest in available diatoms
    FZmu = fieldset.pmZmu * math.pow(particle.mic_zoo, 2)  # (mmol N m-3)**2 Interest in available microzooplankton
    FD = fieldset.pmD * math.pow(particle.detr, 2)  # (mmol N m-3)**2 Interest in available detritus
    Fm = FPn + FPd + FZmu + FD  # (mmol N m-3)**2 Interest in total available food

    GmPd = (fieldset.Gm * fieldset.pmPd * math.pow(particle.d_phy, 2) * particle.mes_zoo) / (
                math.pow(fieldset.km, 2) + Fm)  # [mmol N m-3 s-1]

    gr0 = GmPd
    gr1 = gr0 * wt_N  # conversion to [mg N m-3 s-1]
    gr_n = gr1 / med_N2cell  # conversion to [no. m-3 s-1]
    if ad > 0.:
        gr_ad = gr_n / ad  # conversion to [s-1]
    else:
        gr_ad = 0.

    # ------ Non-linear losses ------
    a_nlin0 = fieldset.mu2 * particle.d_phy * particle.d_phy / (
                fieldset.kPd + particle.d_phy)  # ambient diatom non-linear losses [mmol N m-3 s-1]
    a_nlin1 = a_nlin0 * wt_N  # conversion to [mg N m-3 s-1]
    a_nlin_n = a_nlin1 / med_N2cell  # conversion to [no. m-3 s-1]
    if ad > 0.:
        a_nlin = a_nlin_n / ad  # conversion to [s-1]
    else:
        a_nlin = 0.

    # ------ N:Si ratio density ------
    R_Si_N = particle.d_si / particle.d_phy  # [(mmol N) (mmol Si)-1]

    particle.Si_N = R_Si_N

    rho_bf = fieldset.Rho_bf
    particle.rho_bf = rho_bf

    # ------ Volumes -----
    v_pl = (4. / 3.) * math.pi * particle.r_pl ** 3.  # volume of plastic [m3]
    theta_pl = 4. * math.pi * particle.r_pl ** 2.  # surface area of plastic particle [m2]
    r_a = ((3. / 4.) * (v_a / math.pi)) ** (1. / 3.)  # radius of an algal cell [m]

    v_bfa = (v_a * a) * theta_pl  # volume of living biofilm [m3]
    v_tot = v_bfa + v_pl  # volume of total (biofilm + plastic) [m3]
    t_bf = ((v_tot * (3. / (4. * math.pi))) ** (1. / 3.)) - particle.r_pl  # biofilm thickness [m]

    # ------ Diffusivity -----
    r_tot = particle.r_pl + t_bf  # total radius [m]
    rho_tot = (v_pl * particle.rho_pl + v_bfa * rho_bf) / v_tot  # total density [kg m-3]
    theta_tot = 4. * math.pi * r_tot ** 2.  # surface area of total [m2]
    d_pl = k * (t + 273.16) / (6. * math.pi * sw_visc * r_tot)  # diffusivity of plastic particle [m2 s-1]
    d_a = k * (t + 273.16) / (6. * math.pi * sw_visc * r_a)  # diffusivity of algal cells [m2 s-1]

    # ------ Encounter rates -----
    beta_abrown = 4. * math.pi * (d_pl + d_a) * (r_tot + r_a)  # Brownian motion [m3 s-1]
    beta_ashear = 1.3 * gamma * ((r_tot + r_a) ** 3.)  # advective shear [m3 s-1]
    beta_aset = (1. / 2.) * math.pi * r_tot ** 2. * abs(vs)  # differential settling [m3 s-1]
    beta_a = beta_abrown + beta_ashear + beta_aset  # collision rate [m3 s-1]

    # ------ Attached algal growth (Eq. 11 in Kooi et al. 2017) -----
    a_coll = (beta_a * ad) / theta_pl * fieldset.collision_eff  # [no. m-2 s-1] collisions with diatoms
    a_growth = mu_aa * a

    a_grazing = gr_ad * a
    a_linear = fieldset.mu1 * a  # linear losses [no. m-2 s-1]
    a_non_linear = a_nlin * a  # non-linear losses [no. m-2 s-1]
    a_resp = (q10 ** ((t - 20.) / 10.)) * r20 * a  # [no. m-2 s-1] respiration

    particle.a_coll = a_coll
    particle.a_growth = a_growth

    particle.a_gr = a_grazing
    particle.a_nl = a_non_linear
    particle.a_l = a_linear
    particle.a_resp = a_resp
    particle.a += (a_coll + a_growth - a_grazing - a_resp - a_non_linear) * particle.dt

    dn = 2. * (r_tot)  # equivalent spherical diameter [m]
    delta_rho = (rho_tot - rho_sw) / rho_sw  # normalised difference in density between total plastic+bf and seawater[-]
    dstar = ((rho_tot - rho_sw) * g * dn ** 3.) / (rho_sw * kin_visc ** 2.)  # [-]

    if dstar > 5e9:
        w = 1000.
    elif dstar < 0.05:
        w = (dstar ** 2.) * 1.71E-4
    else:
        w = 10. ** (-3.76715 + (1.92944 * math.log10(dstar)) - (0.09815 * math.log10(dstar) ** 2.) - (
                    0.00575 * math.log10(dstar) ** 3.) + (0.00056 * math.log10(dstar) ** 4.))

    # ------ Settling of particle -----
    if delta_rho > 0:  # sinks
        vs = (g * kin_visc * w * delta_rho) ** (1. / 3.)
    else:  # rises
        a_del_rho = delta_rho * -1.
        vs = -1. * (g * kin_visc * w * a_del_rho) ** (1. / 3.)  # m s-1

    particle.vs_init = vs

    z0 = z + vs * particle.dt
    if z0 <= 0.6 or z0 >= 4000.:  # NEMO's 'surface depth'
        vs = 0
        particle.depth = 0.6
    else:
        particle.depth += vs * particle.dt

    particle.vs = vs
    particle.rho_tot = rho_tot
    particle.r_tot = r_tot
    particle.delta_rho = delta_rho
    particle.t_bf = t_bf

def markov_0_mixing(particle, fieldset, time):
    """
    If a particle tries to cross the boundary, then it stays at the surface
    Author: Victor Onink
    Adapted 1D -> 3D
    Ceiling + tides
    """
    g = fieldset.G
    rho_a = fieldset.Rho_a
    wave_age = fieldset.Wave_age
    phi = fieldset.Phi
    vk = fieldset.Vk

    rho_sw = particle.density
    mld = fieldset.mldr[time, particle.depth, particle.lat, particle.lon]
    particle.tau = fieldset.taum[time, particle.depth, particle.lat, particle.lon]
    particle.mld = mld
    particle.w10 = fieldset.w_10[time, particle.depth, particle.lat, particle.lon]

    # Define KPP profile from tau and mld
    u_s_a = math.sqrt(particle.tau / rho_a)
    u_s_w = math.sqrt(particle.tau / rho_sw)

    alpha_dt = (vk * u_s_w) / (phi * mld ** 2)
    alpha = (vk * u_s_w) / phi

    beta = wave_age * u_s_a / particle.w10
    z0 = 3.5153e-5 * math.pow(beta, -0.42) * math.pow(particle.w10, 2) / g

    if particle.depth<mld:
        dK_z = alpha_dt * (mld - particle.depth) * (mld - 3 * particle.depth - 2 * z0)
    else:
        dK_z = 0

    particle.KPP = alpha * (particle.depth + 0.5 * dK_z * particle.dt + z0) * math.pow(
        1 - (particle.depth + 0.5 * dK_z * particle.dt) / mld, 2)
    if particle.depth<mld:
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
    particle.w_m = w_m_step / particle.dt

    # The ocean surface acts as a lid off of which the plastic bounces if tries to cross the ocean surface
    potential = particle.depth + w_m_step
    if potential < 0.6:
        particle.depth = 0.6
    else:
        particle.depth = potential

def AdvectionRK4_1D(particle, fieldset, time):
    """Advection of particles using fourth-order Runge-Kutta integration including vertical velocity.
    Function needs to be converted to Kernel object before execution"""
    (w1) = fieldset.W[time, particle.depth, particle.lat, particle.lon]
    # lon1 = particle.lon + u1*.5*particle.dt
    # lat1 = particle.lat + v1*.5*particle.dt
    dep1 = particle.depth + w1 * .5 * particle.dt
    (w2) = fieldset.W[time + .5 * particle.dt, dep1, particle.lat, particle.lon]
    # lon2 = particle.lon + u2*.5*particle.dt
    # lat2 = particle.lat + v2*.5*particle.dt
    dep2 = particle.depth + w2 * .5 * particle.dt
    (w3) = fieldset.W[time + .5 * particle.dt, dep2, particle.lat, particle.lon]
    # lon3 = particle.lon + u3*particle.dt
    # lat3 = particle.lat + v3*particle.dt
    dep3 = particle.depth + w3 * particle.dt
    (w4) = fieldset.W[time + particle.dt, dep3, particle.lat, particle.lon]
    # particle.lon += particle.lon #(u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
    # particle.lat += particle.lat #lats[1,1] #(v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt
    particle.depth += (w1 + 2 * w2 + 2 * w3 + w4) / 6. * particle.dt