import os
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import cpu_count

from mrexo import fit_mr_relation


pwd = os.path.dirname(__file__)

#pwd = '~/mrexo_working/'

t = Table.read(os.path.join(pwd,'Cool_stars_20181211_exc_upperlim.csv'))

# Symmetrical errorbars
Mass_sigma = (abs(t['pl_masseerr1']) + abs(t['pl_masseerr2']))/2
Radius_sigma = (abs(t['pl_radeerr1']) + abs(t['pl_radeerr2']))/2

# In Earth units
Mass = np.array(t['pl_masse'])
Radius = np.array(t['pl_rade'])

logMass = np.log10(Mass)
logRadius = np.log10(Radius)
logMass_sigma = 0.434 * Mass_sigma/Mass
logRadius_sigma = 0.434 * Radius_sigma/Radius
'''
plt.plot(logRadius,logMass, '.')
plt.show()


'''

p = np.poly1d(np.polyfit(logRadius, logMass, 1))

R_min = np.min(Radius)*1
R_max = np.max(Radius)*1


sim_sizes = [10,20,40,50,60,80,100]

for i in sim_sizes:
    data_size = i

    sim_radius = np.linspace(np.log10(R_min),np.log10(R_max), data_size)
    sim_mass = 10**p(sim_radius)
    sim_radius = 10**sim_radius
    sim_radius_error = 0.1 * sim_radius
    sim_mass_error = 0.1 * sim_mass
    

    # Directory to store results in
    result_dir = os.path.join(pwd)
    
    if __name__ == '__main__':
        initialfit_result, bootstrap_results = fit_mr_relation(Mass = sim_mass, Mass_sigma = sim_mass_error,
                                                Radius = sim_radius, Radius_sigma = sim_radius_error,
                                                save_path = os.path.join(result_dir,'simulation_{}_points'.format(data_size)), select_deg = 'cv',
                                                num_boot = 30, cores = cpu_count()-2)
