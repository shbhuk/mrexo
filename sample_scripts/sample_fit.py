import os
from astropy.table import Table
import numpy as np
from multiprocessing import cpu_count
import numpy as np


from mrexo import fit_xy_relation


try :
    pwd = os.path.dirname(__file__)
except NameError:
    pwd = ''
    print('Could not find pwd')



'''
Sample script to fit mass-radius relationship.
The CSV table is generated from the NASA Exoplanet Archive. The existing example
is for the 24 M dwarf planets as explained in Kanodia 2019.
This can be replaced with any other CSV file.

For this sample, the cross validation has already been performed and the optimum number of
degrees has been established to be 17. For a new sample, set select_deg = 'cv' to
use cross validation to find the optimum number of degrees.

Can use parallel processing by setting cores > 1.
To use all the cores in the CPU, cores=cpu_count() (from multiprocessing import cpu_count)

To bootstrap and estimate the robustness of the median, set num_boot > 1.
If cores > 1, then uses parallel processing to run the various boots. For large datasets,
first run with num_boot to be a smaller number to estimate the computational time.

For more detailed guidelines read the docuemtnation for the fit_mr_relation() function.
'''


t = Table.read(os.path.join(pwd,'Cool_stars_MR_20200116_exc_upperlim.csv'))

# Symmetrical errorbars
Mass_sigma = (abs(t['pl_masseerr1']) + abs(t['pl_masseerr2']))/2
Radius_sigma = (abs(t['pl_radeerr1']) + abs(t['pl_radeerr2']))/2

# In Earth units
Mass = np.array(t['pl_masse'])
Radius = np.array(t['pl_rade'])

# Directory to store results in
result_dir = os.path.join(pwd,'M_dwarfs_20200116')

# Run with 100 bootstraps. Selecting degrees to be 17. Alternatively can set select_deg = 'cv' to
# find the optimum number of degrees.

RadiusDict = {'X': Radius, 'X_sigma': Radius_sigma, 'X_max':None, 'X_min':None, 'X_label':'Radius', 'X_char':'r'}
MassDict = {'Y': Mass, 'Y_sigma': Mass_sigma, 'Y_max':None, 'Y_min':None, 'Y_label':'Mass', 'Y_char':'m'}

if __name__ == '__main__':
            initialfit_result, bootstrap_results = fit_xy_relation(**RadiusDict, **MassDict,
                                                save_path = result_dir, select_deg = "cv",
                                                num_boot = 100, cores = 2)
