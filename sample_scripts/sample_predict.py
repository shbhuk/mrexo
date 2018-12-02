import matplotlib.pyplot as plt
import os
from astropy.table import Table
import numpy as np

from mrexo import predict_m_given_r, predict_r_given_m


try :
    pwd = os.path.dirname(__file__)
except NameError:
    pwd = ''

result_dir = os.path.join(pwd,'Results_deg_11')
#result_dir = "C:/Users/shbhu/Documents/Git/mrexo/sample_kepler2/Kepler_55_run"
#weights_mle = np.loadtxt(os.path.join(result_dir,'output','weights.txt'))

a = predict_m_given_r(Radius = 1., Radius_sigma = None, posterior_sample = False, islog = False, result_dir = result_dir)

b = predict_r_given_m(Mass = a[0], Mass_sigma = None, posterior_sample = False, islog = False, result_dir = result_dir)
#print(predict_m_given_r(Radius = 1., Radius_sigma = 0.1, posterior_sample = False, islog = True, weights_mle = weights_mle))

print(a,b)
