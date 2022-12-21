import os
from astropy.table import Table
import numpy as np
from multiprocessing import cpu_count
import numpy as np


from mrexo import fit_xy_relation
from mrexo import predict_from_measurement, plot_joint_xy_distribution, plot_mle_weights, plot_y_given_x_relation
from mrexo.mle_utils import calc_C_matrix, optimizer, rank_FI_matrix
import pandas as pd

try :
    pwd = os.path.dirname(__file__)
except NameError:
    pwd = ''
    print('Could not find pwd')




t = Table.read(os.path.join(pwd,'Cool_stars_20200520_exc_upperlim.csv'))
# t = Table.read(os.path.join(pwd,'Kepler_MR_inputs.csv'))
t = Table.read(os.path.join(pwd,'FGK_20190406.csv'))

# Symmetrical errorbars
Mass_sigma = (abs(t['pl_masseerr1'])) #+ abs(t['pl_masseerr2']))/2
Radius_sigma = (abs(t['pl_radeerr1']))# + abs(t['pl_radeerr2']))/2

# In Earth units
Mass = np.array(t['pl_masse'])
Radius = np.array(t['pl_rade'])

# Directory to store results in
# result_dir = os.path.join(pwd,'Mdwarfs_20200520_cv50')
result_dir = os.path.join(pwd,'FGK319_TestFI')
# result_dir = os.path.join(pwd, 'FGK_319_cv100')

# Run with 100 bootstraps. Selecting degrees to be 17. Alternatively can set select_deg = 'cv' to
# find the optimum number of degrees.

RadiusDict = {'X': Radius, 'X_sigma': Radius_sigma, 'X_max':None, 'X_min':None, 'X_label':'Radius', 'X_char':'r'}
MassDict = {'Y': Mass, 'Y_sigma': Mass_sigma, 'Y_max':None, 'Y_min':None, 'Y_label':'Mass', 'Y_char':'m'}

deg=55

C_pdf = calc_C_matrix(n=len(Mass_sigma), deg=deg, \
    Y=Mass, Y_sigma=Mass_sigma, Y_max=3.5549, Y_min=-2, \
    X=Radius, X_sigma=Radius_sigma, X_max=1.399, X_min=-0.215, \
    abs_tol=1e-8, save_path=result_dir, Log=False, verbose=2, SaveCMatrix=True)


unpadded_weight, n_log_lik = optimizer(C_pdf=C_pdf, deg=deg,
                verbose=2, save_path=result_dir)

w_sq = np.reshape(unpadded_weight,[deg-2,deg-2])
w_sq_padded = np.zeros((deg,deg))
w_sq_padded[1:-1,1:-1] = w_sq
w_hat = w_sq_padded.flatten()

np.savetxt(os.path.join(result_dir, 'unpaddedweights.txt'), unpadded_weight)
np.savetxt(os.path.join(result_dir, 'weights.txt'), w_hat)

start = datetime.datetime.now()
Rank = rank_FI_matrix(C_pdf, unpadded_weight)
end = datetime.datetime.now()

print(end-start)
