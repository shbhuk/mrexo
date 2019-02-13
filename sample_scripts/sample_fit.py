import os
from astropy.table import Table
import numpy as np
from multiprocessing import cpu_count
import numpy as np

from scipy.stats.mstats import mquantiles
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from mrexo.mle_utils import cond_density_quantile
from mrexo.utils import load_lookup_table
from mrexo.plot import plot_r_given_m_relation, plot_m_given_r_relation

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
result_dir = os.path.join(pwd,'M_dwarfs_cv')

# Run with 100 bootstraps. Selecting degrees to be 17. Alternatively can set select_deg = 'cv' to
# find the optimum number of degrees.

if __name__ == '__main__':
            initialfit_result, bootstrap_results = fit_mr_relation(Mass = Mass, Mass_sigma = Mass_sigma,
                                                Radius = Radius, Radius_sigma = Radius_sigma,
                                                save_path = result_dir, select_deg = 17,
                                                num_boot = 100, cores = cpu_count())
