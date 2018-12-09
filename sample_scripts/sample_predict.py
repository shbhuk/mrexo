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
result_dir = os.path.join(pwd,'M_dwarfs_11')
result_dir = "C:/Users/shbhu/Documents/Git/mrexo/sample_kepler2/Kepler_55_open_corrected"
#weights_mle = np.loadtxt(os.path.join(result_dir,'output','weights.txt'))
a=1
#a = predict_m_given_r(Radius = np.log10(5), Radius_sigma = None, posterior_sample = False, islog = True, dataset = 'mdwarf')
#b = predict_m_given_r(Radius = 1, Radius_sigma = 0.1, posterior_sample = False, islog = True, dataset = 'kepler', showplot = True)

#c = predict_r_given_m(Mass = b[0], Mass_sigma = None, posterior_sample = False, islog = True, dataset = 'kepler', showplot = True)
#print(predict_m_given_r(Radius = 1., Radius_sigma = 0.1, posterior_sample = False, islog = True, dataset = 'Kepler'))

b = predict_m_given_r(Radius = np.linspace(1.,1.2,10), Radius_sigma = np.repeat(0.1,10), posterior_sample = True, islog = False, dataset = 'mdwarf', showplot = True)
