import os
from astropy.table import Table
import numpy as np

from mrexo import fit_mr_relation



t = Table.read(os.path.join(os.path.dirname(__file__),'Cool_stars_20181109.csv'))
t = t.filled()

Mass_sigma = (abs(t['pl_masseerr1']) + abs(t['pl_masseerr2']))/2
Radius_sigma = (abs(t['pl_radeerr1']) + abs(t['pl_radeerr2']))/2

Mass = np.array(t['pl_masse'])
Radius = np.array(t['pl_rade'])

# bounds for Mass and Radius
Radius_min = -0.3
Radius_max = np.log10(max(Radius) + np.std(Radius)/np.sqrt(len(Radius)))
Radius_max = np.log10(max(Radius + Radius_sigma))
Mass_min = np.log10(max(min(Mass) - np.std(Mass)/np.sqrt(len(Mass)), 0.01))
Mass_max = np.log10(max(Mass) + np.std(Mass)/np.sqrt(len(Mass)))
num_boot = 10



run_degrees = np.arange(0,10)


if __name__ == '__main__':
    a = fit_mr_relation(Mass = Mass, Mass_sigma = Mass_sigma, Radius = Radius, Radius_sigma = Radius_sigma, Mass_max = Mass_max,
                        Mass_min = Mass_min, Radius_max = Radius_max, Radius_min = Radius_min, degree_max = 30, select_deg = 11, Log = True, num_boot = num_boot,
                        location = os.path.join(os.path.dirname(__file__),'M_dwarfs_deg_{}'.format(11)), abs_tol = 1e-10)
