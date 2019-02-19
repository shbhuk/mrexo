import os
from astropy.table import Table
import numpy as np
from multiprocessing import cpu_count
import numpy as np
import cProfile
from scipy.stats import norm,beta
import scipy
import datetime as datetime
from mrexo import MLE_fit
import timeit

from mrexo import fit_mr_relation, predict_from_measurement


try :
    pwd = os.path.dirname(__file__)
except NameError:
    pwd = ''
    print('Could not find pwd')

t = Table.read(os.path.join(pwd,'Cool_stars_MR_20181214_exc_upperlim.csv'))

# Symmetrical errorbars
Mass_sigma = (abs(t['pl_masseerr1']) + abs(t['pl_masseerr2']))/2
Radius_sigma = (abs(t['pl_radeerr1']) + abs(t['pl_radeerr2']))/2

# In Earth units
Mass = np.array(t['pl_masse'])
Radius = np.array(t['pl_rade'])

# Directory to store results in
result_dir = os.path.join(pwd,'M_dwarfs_dummy')
print(1)

Mass_min = np.log10(max(min(Mass - Mass_sigma), 0.01))
Mass_max = np.log10(max(Mass + Mass_sigma))
Radius_min = min(np.log10(min(np.abs(Radius - Radius_sigma))), -0.3)
Radius_max = np.log10(max(Radius + Radius_sigma))

Mass_bounds = np.array([Mass_min, Mass_max])
Radius_bounds = np.array([Radius_min, Radius_max])

# result = MLE_fit(Mass=Mass, Radius=Radius, Mass_sigma=Mass_sigma, Radius_sigma=Radius_sigma,Mass_bounds=Mass_bounds, Radius_bounds=Radius_bounds,  deg=17, abs_tol=1e-8, save_path=pwd, calc_joint_dist = True, output_weights_only=False)

start = datetime.datetime.now()
cProfile.run('MLE_fit(Mass=Mass, Radius=Radius, Mass_sigma=Mass_sigma, Radius_sigma=Radius_sigma,Mass_bounds=Mass_bounds, Radius_bounds=Radius_bounds,  deg=17, abs_tol=1e-8, save_path=pwd, calc_joint_dist = True, output_weights_only=False)', 'mle_profile')

# cProfile.run('predict_from_measurement(measurement=1, measurement_sigma=0.25, result_dir=result_dir, use_lookup = False, qtl = [0.3,0.2,0.1,0.5,0.2,0.6])', 'predict_profile')
end = datetime.datetime.now()

import pstats
p = pstats.Stats('predict_profile')
# p.strip_dirs().sort_stats(-1).print_stats()
p.sort_stats('time').print_stats(10)

print(end-start)



"""
#Factorial

import math
import scipy

def read_factorial(n):
    if n == 0:
        return 1
    else:
        return n * read_factorial(n-1)

def factorial(n):return reduce(lambda x,y:x*y,[1]+range(1,n+1))


print(timeit.timeit('"factorial(42)"', number = int(1e8)))
print(timeit.timeit('"read_factorial(42)"', number = int(1e8)))
print(timeit.timeit('"math.factorial(42)"', number = int(1e8)))
print(timeit.timeit('"scipy.math.factorial(42)"', number = int(1e8)))
"""






"""

# Comparing PDFs

def norm_pdf(x, loc, scale):
    '''
    Find the PDF for a normal distribution. Identical to scipy.stats.norm.pdf.
    Runs much quicker without the generic function handling.
    '''
    y = (x - loc)/scale
    return np.exp(-y*y/2)/(np.sqrt(2*np.pi))/scale

def int_gamma(a):
    return scipy.math.factorial(a-1)


def beta_pdf(x,a,b):
    f = (int_gamma(a+b) * x**(a-1)*(1-x)**(b-1))/(int_gamma(a)*int_gamma(b))
    return f




# start = datetime.datetime.now()


for i in range(0,100):
    # norm.pdf(x = 1, loc = 0, scale = 1)
    x = np.random.uniform(0,1)
    print(x,beta.pdf(x,10,5) - beta_pdf(x,10,5))

    loc = 0.32
    scale = 0.5
    print(norm.pdf(x=x, loc=loc, scale=scale) - norm_pdf(x=x, loc=loc, scale=scale))
    print('')


# print(datetime.datetime.now() - start)


x = 1


print(norm.pdf(x = x, loc = loc, scale = scale))
print(norm_pdf(x = x, loc = loc, scale = scale))
"""
