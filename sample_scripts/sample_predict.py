import matplotlib.pyplot as plt
import os
from astropy.table import Table
import numpy as np

from mrexo import predict_m_given_r, predict_r_given_m


try :
    pwd = os.path.dirname(__file__)
except NameError:
    pwd = ''



result_dir = "C:/Users/shbhu/Documents/Git/mrexo/sample_kepler2/Kepler_55_open_corrected"


a = predict_m_given_r(Radius=2.5, Radius_sigma = 0.3, posterior_sample=False, islog=False, dataset = 'mdwarf', showplot=True)
print(a)
c = predict_r_given_m(Mass=3, Mass_sigma=None, posterior_sample=False, islog=False, dataset = 'mdwarf', showplot=True)
# print(a,c)
#b = predict_m_given_r(Radius=1, Radius_sigma=0.1, posterior_sample=False, islog=True, dataset='mdwarf', showplot=True)

'''
C:/Users/shbhu/Documents/Git/mrexo/sample_scripts/M_dwarfs_deg17
17
[1.9423930353731744, 0.4405540095682138, 8.992463492366237]

'''
