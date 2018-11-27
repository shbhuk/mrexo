import os
from astropy.table import Table
import numpy as np

from mrexo import fit_mr_relation

try :
    pwd = os.path.dirname(__file__)
except NameError:
    pwd = ''

t = Table.read(os.path.join(pwd,'Cool_stars_20181109.csv'))

# Symmetrical errorbars
Mass_sigma = (abs(t['pl_masseerr1']) + abs(t['pl_masseerr2']))/2
Radius_sigma = (abs(t['pl_radeerr1']) + abs(t['pl_radeerr2']))/2

# In Earth units
Mass = np.array(t['pl_masse'])
Radius = np.array(t['pl_rade'])

# Bounds for Mass and Radius in log10()
Radius_min = -0.3
Radius_max = np.log10(max(Radius + Radius_sigma))
Mass_min = np.log10(max(min(Mass) - np.std(Mass)/np.sqrt(len(Mass)), 0.01))
Mass_max = np.log10(max(Mass) + np.std(Mass)/np.sqrt(len(Mass)))

# Number of bootstraps to run for
num_boot = 50

# The degrees for the Bernstein polynomials
select_deg = 11

select_deg = 'cv'


# Directory to store results in 
result_dir = os.path.join(pwd,'M_dwarfs_deg_{}'.format(select_deg))

os.mkdir(result_dir)
copyfile()

if __name__ == '__main__':
    a = fit_mr_relation(Mass = Mass, Mass_sigma = Mass_sigma, 
                        Radius = Radius, Radius_sigma = Radius_sigma,
                        Mass_max = Mass_max, Mass_min = Mass_min,
                        Radius_max = Radius_max, Radius_min = Radius_min, 
                        degree_max = 30, select_deg = select_deg, 
                        num_boot = num_boot,
                        location = result_dir, abs_tol = 1e-10)
