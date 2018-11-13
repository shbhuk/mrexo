import os
import sys
import datetime
from astropy.table import Table
import numpy as np

sys.path.append(os.path.dirname(__file__))

from working_Mdwarfs import MLE_fit_bootstrap



t = Table.read(os.path.join(os.path.dirname(__file__),'Cool_stars_20181109.csv'))
t = t.filled()

M_sigma = (abs(t['pl_masseerr1']) + abs(t['pl_masseerr2']))/2
R_sigma = (abs(t['pl_radeerr1']) + abs(t['pl_radeerr2']))/2

M_obs = np.array(t['pl_masse'])
R_obs = np.array(t['pl_rade'])

# bounds for Mass and Radius
Radius_min = -0.3
Radius_max = np.log10(max(R_obs) + np.std(R_obs)/np.sqrt(len(R_obs)))
Radius_max = np.log10(max(R_obs + R_sigma))
#Radius_max = 1.4
Mass_min = np.log10(max(min(M_obs) - np.std(M_obs)/np.sqrt(len(M_obs)), 0.01))
Mass_max = np.log10(max(M_obs) + np.std(M_obs)/np.sqrt(len(M_obs)))
num_boot = 100


Mass = M_obs
Radius = R_obs
Mass_sigma = np.array(M_sigma)
Radius_sigma = np.array(R_sigma)
Mass_max = Mass_max
Mass_min = Mass_min
Radius_max = Radius_max
Radius_min = Radius_min



run_degrees = np.arange(0,10)

for d in run_degrees:
    if __name__ == '__main__':
        a = MLE_fit_bootstrap(Mass = M_obs, Radius = R_obs, Mass_sigma = M_sigma, Radius_sigma = R_sigma, Mass_max = Mass_max,
                            Mass_min = Mass_min, Radius_max = Radius_max, Radius_min = Radius_min, degree_max = 30, select_deg = 'cv', Log = True, num_boot = 100,
                            location = os.path.join(os.path.dirname(__file__),'M_dwarfs_CV_{}'.format(d)), abs_tol = 1e-10)
                        
    
    
