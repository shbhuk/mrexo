import matplotlib.pyplot as plt
import os
from astropy.table import Table
import numpy as np

from mrexo import predict_m_given_r


pwd = os.path.dirname(__file__)


result_dir = os.path.join(pwd,'M_dwarfs_deg_11')


weights_mle = np.loadtxt(os.path.join(result_dir,'weights.txt'))

print(predict_m_given_r(Radius = 1., Radius_sigma = None, posterior_sample = False, islog = True, weights_mle = weights_mle))
print(predict_m_given_r(Radius = 1., Radius_sigma = 0.1, posterior_sample = False, islog = True, weights_mle = weights_mle))
