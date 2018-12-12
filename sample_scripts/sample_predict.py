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
#result_dir = "C:/Users/shbhu/Documents/GitHub/mrexo/sample_kepler2/Kepler_55_cluster"

#weights_mle = np.loadtxt(os.path.join(result_dir,'output','weights.txt'))
a=1
#a = predict_m_given_r(Radius=1.64, Radius_sigma=None, posterior_sample=False, islog=False, result_dir=result_dir, showplot=True)
print(a)
#b = predict_m_given_r(Radius=1, Radius_sigma=0.1, posterior_sample=False, islog=True, dataset='kepler', showplot=True)

c = predict_r_given_m(Mass=np.log10(1), Mass_sigma=None, posterior_sample=False, islog=True, dataset='kepler', showplot=True)
import datetime
print(datetime.datetime.now())
print(predict_r_given_m(Mass=np.linspace(1.,1.2,10), Mass_sigma=np.repeat(0.1,10), posterior_sample=True, islog=True, dataset='kepler', showplot=True))
print(datetime.datetime.now())
#b = predict_m_given_r(Radius=np.linspace(1.,1.2,10), Radius_sigma=np.repeat(0.1,10), posterior_sample=True, islog=False, dataset='mdwarf', showplot=True)
