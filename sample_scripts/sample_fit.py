import os
from astropy.table import Table
import numpy as np
from multiprocessing import cpu_count
import numpy as np


from mrexo import fit_xy_relation
from mrexo import predict_from_measurement
import pandas as pd

try :
    pwd = os.path.dirname(__file__)
except NameError:
    pwd = 'C:\\\\Users\\\\shbhu\\\\Documents\\\\GitHub\\\\mrexo\\\\sample_scripts'
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


t = Table.read(os.path.join(pwd,'Cool_stars_20200520_exc_upperlim.csv'))
# t = Table.read(os.path.join(pwd,'Kepler_MR_inputs.csv'))
# t = Table.read(os.path.join(pwd,'FGK_20190406.csv'))

# Symmetrical errorbars
Mass_sigma1 = abs(t['pl_masseerr1'])
Mass_sigma2 = abs(t['pl_masseerr1'])
Radius_sigma1 = abs(t['pl_radeerr1'])
Radius_sigma2 = abs(t['pl_radeerr1'])


# In Earth units
Mass = np.array(t['pl_masse'])
Radius = np.array(t['pl_rade'])
Period = np.array(t['pl_orbper'])

Period_sigma = np.repeat(np.nan, len(Period))

# Directory to store results in
# result_dir = os.path.join(pwd,'Mdwarfs_20200520_cv50')
result_dir = os.path.join(pwd,'Kepler127_aic')
# result_dir = os.path.join(pwd, 'FGK_319_cv100')

# Run with 100 bootstraps. Selecting degrees to be 17. Alternatively can set select_deg = 'cv' to
# find the optimum number of degrees.

# RadiusDict = {'X': Radius, 'X_sigma': Radius_sigma, 'X_max':None, 'X_min':None, 'X_label':'Radius', 'X_char':'r'}
# MassDict = {'Y': Mass, 'Y_sigma': Mass_sigma, 'Y_max':None, 'Y_min':None, 'Y_label':'Mass', 'Y_char':'m'}


FakePeriod = np.ones(len(Period))
FakePeriodSigma = FakePeriod*0.01
Period_sigma = FakePeriodSigma

RadiusDict = {'Data': Radius, 'SigmaLower': Radius_sigma1,  "SigmaUpper":Radius_sigma2, 'Max':np.nan, 'Min':np.nan, 'Label':'Radius', 'Char':'r'}
MassDict = {'Data': Mass, 'SigmaLower': Mass_sigma1, "SigmaUpper":Mass_sigma2, 'Max':np.nan, 'Min':np.nan, 'Label':'Mass', 'Char':'m'}
PeriodDict = {'Data': FakePeriod, 'SigmaLower': Period_sigma, "SigmaUpper":Period_sigma, 'Max':np.nan, 'Min':np.nan, 'Label':'Period', 'Char':'p'}

from mrexo.mle_utils_nd import InputData, MLE_fit, _find_indv_pdf
import matplotlib.pyplot as plt
DataDict = InputData([MassDict, RadiusDict])
save_path = 'C:\\Users\\shbhu\\Documents\\GitHub\\mrexo\\sample_scripts\\Trial_nd'
 
ndim = len(DataDict)
deg_per_dim = [24, 24]

outputs = MLE_fit(DataDict, 
	deg_per_dim=deg_per_dim,
	save_path=save_path, output_weights_only=False, calc_joint_dist=False)




x = outputs['DataSequence'][0]
y = outputs['DataSequence'][1]
# z = outputs['DataSequence'][2]
JointDist = outputs['JointDist']
weights = outputs['Weights']

i = 10
"""
plt.imshow(JointDist, extent=(x.min(), x.max(), y.min(), y.max()), aspect='auto', origin='lower'); 
plt.plot(np.log10(Mass), np.log10(Radius),  'k.')
# plt.title("Orbital Period = {} d".format(str(np.round(title,3))))
plt.ylabel("Log10 Radius");
plt.xlabel("Log10 Mass");
plt.tight_layout()
plt.show(block=False)
"""

from mrexo.mle_utils_nd import cond_density_quantile

results = cond_density_quantile(a=np.log10(4),
	a_min=-0.173, a_max=1.37, 
	b_max=2.487, b_min=-0.627, 
	deg=24, deg_vec=np.arange(1,25), w_hat=weights)


"""
if __name__ == '__main__':
            initialfit_result, _ = fit_xy_relation(**RadiusDict, **MassDict,
                                                save_path = result_dir, select_deg = 'aic',
                                                num_boot = 100, cores = 4, degree_max=100)

"""
