import matplotlib.pyplot as plt
import os
from astropy.table import Table
import numpy as np

from mrexo import predict_m_given_r


try :
    pwd = os.path.dirname(__file__)
except NameError:
    pwd = ''

result_dir = os.path.join(pwd,'M_dwarfs_deg_cv')
weights_mle = np.loadtxt(os.path.join(result_dir,'output','weights.txt'))

a = predict_m_given_r(Radius = 1, Radius_sigma = None, posterior_sample = False, islog = True)
#print(predict_m_given_r(Radius = 1., Radius_sigma = 0.1, posterior_sample = False, islog = True, weights_mle = weights_mle))
