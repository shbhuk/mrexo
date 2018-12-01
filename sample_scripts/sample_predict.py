import matplotlib.pyplot as plt
import os
from astropy.table import Table
import numpy as np

from mrexo import predict_m_given_r


try :
    pwd = os.path.dirname(__file__)
except NameError:
    pwd = ''

result_dir = os.path.join(pwd,'Results_deg_11')
#result_dir = "C:/Users/shbhu/Documents/Git/mrexo/sample_kepler2/Kepler_55_run"
#weights_mle = np.loadtxt(os.path.join(result_dir,'output','weights.txt'))

print(predict_m_given_r(Radius = 1., Radius_sigma = None, posterior_sample = False, islog = False, result_dir = result_dir))
#print(predict_m_given_r(Radius = 1., Radius_sigma = 0.1, posterior_sample = False, islog = True, weights_mle = weights_mle))


'''
1,0.1, log = True

11
0.3327554114474372,
[1.9431821836263885, 1.4909558986255853, 2.369607462654234]

1, None, log = True
11
0.325524187669665
[1.9957103093255821, 1.6168196191199837, 2.374543227450403]


1, None, log = False
11
1.0922766223502307
[1.0561056847353643, 0.22730647510097013, 4.980641827384585]


1, 0.1, log = False
11
1.0211018676806611
[1.0628849396460824, 0.2136666306351997, 5.3669104641223075]
'''
