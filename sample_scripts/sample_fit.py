import os
from astropy.table import Table
import numpy as np
from multiprocessing import cpu_count
import numpy as np


from mrexo import fit_mr_relation


try :
    pwd = os.path.dirname(__file__)
except NameError:
    pwd = ''
    print('Could not find pwd')

t = Table.read(os.path.join(pwd,'Cool_stars_MR_20181214_exc_upperlim.csv'))

# Symmetrical errorbars
Mass_sigma = (abs(t['pl_masseerr1']) + abs(t['pl_masseerr2']))/2
Radius_sigma = (abs(t['pl_radeerr1']) + abs(t['pl_radeerr2']))/2

# In Earth units
Mass = np.array(t['pl_masse'])
Radius = np.array(t['pl_rade'])

# Directory to store results in
result_dir = os.path.join(pwd,'M_dwarfs_new_cv')

# Run with 100 bootstraps. Selecting degrees to be 17. Alternatively can set select_deg = 'cv' to
# find the optimum number of degrees.

if __name__ == '__main__':
            initialfit_result, bootstrap_results = fit_mr_relation(Mass = Mass, Mass_sigma = Mass_sigma,
                                                Radius = Radius, Radius_sigma = Radius_sigma,
                                                save_path = result_dir, select_deg = 'cv',
                                                num_boot = 100, cores = cpu_count())
