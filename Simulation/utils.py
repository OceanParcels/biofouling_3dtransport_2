import numpy as np

def delete_particle(particle, fieldset, time):
    """Kernel for deleting particles if they are out of bounds."""
    print('particle is deleted out of bounds at lon = ' + str(particle.lon) + ', lat =' + str(
        particle.lat) + ', depth =' + str(particle.depth))
    particle.delete()


def delete_particle_interp(particle, fieldset, time):
    """Kernel for deleting particles if they are out of bounds."""
    print('particle is deleted due to an interpolation error at lon = ' + str(particle.lon) + ', lat =' + str(
        particle.lat) + ', depth =' + str(particle.depth))
    particle.delete()

def periodicBC(particle, fieldset, time):
    if particle.lon <= -180.:
        particle.lon += 360.
    elif particle.lon >= 180.:
        particle.lon -= 360.

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
    r_pls = np.tile(location_r_pls, [n_locs, 1])
    return r_pls

def getclosest_ij(lats, lons, latpt, lonpt):
    """Function to find the index of the closest point to a certain lon/lat value."""
    dist_sq = (lats - latpt) ** 2 + (lons - lonpt) ** 2  # find squared distance of every point on grid
    minindex_flattened = dist_sq.argmin()  # 1D index of minimum dist_sq element
    return np.unravel_index(minindex_flattened,
                            lats.shape)  # Get 2D index for latvals and lonvals arrays from 1D index