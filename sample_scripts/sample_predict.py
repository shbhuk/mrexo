import matplotlib.pyplot as plt
import os
from astropy.table import Table
import numpy as np

from mrexo import predict_m_given_r, predict_r_given_m


try :
    pwd = os.path.dirname(__file__)
except NameError:
    pwd = ''

result_dir = os.path.join(pwd,'M_dwarfs_deg_cv')
weights_mle = np.loadtxt(os.path.join(result_dir,'output','weights.txt'))

predicted_mass, lower_mass, upper_mass  = predict_m_given_r(Radius = 10, Radius_sigma = None, posterior_sample = False, islog = False)
print(predicted_mass, lower_mass, upper_mass)

predicted_radius, lower_radius, upper_radius  = predict_r_given_m(Mass = predicted_mass, Mass_sigma = None, posterior_sample = False, islog = False)
print(predicted_radius, lower_radius, upper_radius)