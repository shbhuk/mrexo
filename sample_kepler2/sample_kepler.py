import os
from astropy.table import Table
import numpy as np
from multiprocessing import cpu_count

from mrexo import fit_mr_relation


pwd = os.path.dirname(__file__)

t = Table.read(os.path.join(pwd,'MR_Kepler_170605_noanalytTTV_noupplim.csv'))

# Symmetrical errorbars
Mass_sigma = (abs(t['pl_masseerr1']) + abs(t['pl_masseerr2']))/2 * 0
Radius_sigma = (abs(t['pl_radeerr1']) + abs(t['pl_radeerr2']))/2 * 0

Mass_sigma = np.repeat(None, len(Mass_sigma))
Radius_sigma = np.repeat(None, len(Mass_sigma))


# In Earth units
Mass = np.array(t['pl_masse'])
Radius = np.array(t['pl_rade'])

Mass_min = -1
Mass_max = 3.80957
Radius_min = -0.3
Radius_max = 1.357509 

# Directory to store results in 
result_dir = os.path.join(pwd,'Kepler_no_error')

if __name__ == '__main__':
    initialfit_result, bootstrap_results = fit_mr_relation(Mass=Mass, Mass_sigma=Mass_sigma,
                                            Radius=Radius, Radius_sigma=Radius_sigma,
                                            Mass_min=Mass_min, Mass_max=Mass_max,
                                            Radius_min=Radius_min, Radius_max=Radius_max,
                                            save_path=result_dir, select_deg=55, 
                                            num_boot=50, cores=cpu_count())
