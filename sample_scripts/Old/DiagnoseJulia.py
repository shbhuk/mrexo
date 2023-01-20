import os
from astropy.table import Table
import numpy as np
from multiprocessing import cpu_count
import numpy as np


from mrexo import fit_xy_relation
from mrexo.mle_utils_beta import calc_C_matrix

try :
    pwd = os.path.dirname(__file__)
except NameError:
    pwd = ''
    print('Could not find pwd')

t = Table.read(os.path.join(pwd,'Cool_stars_MR_20181214_exc_upperlim.csv'))

# Symmetrical errorbars
Mass_sigma = (abs(t['pl_masseerr1'])) #+ abs(t['pl_masseerr2']))/2
Radius_sigma = (abs(t['pl_radeerr1']))# + abs(t['pl_radeerr2']))/2

# In Earth units
Mass = np.array(t['pl_masse'])
Radius = np.array(t['pl_rade'])

RadiusDict = {'X': Radius, 'X_sigma': Radius_sigma, 'X_max':None, 'X_min':None, 'X_label':'Radius', 'X_char':'r'}
MassDict = {'Y': Mass, 'Y_sigma': Mass_sigma, 'Y_max':None, 'Y_min':None, 'Y_label':'Mass', 'Y_char':'m'}

C_pdf = calc_C_matrix(n=24, deg=17, Y=Mass, Y_sigma=Mass_sigma, Y_max=2.447909601958097703, Y_min=-1.744727494896693765e+00, 
    X=Radius, X_sigma=Radius_sigma, X_max=1.3048350692290063, X_min=-0.3, save_path=pwd, Log=True, verbose=True, abs_tol=1e-8)
    
print(np.sum(C_pdf))