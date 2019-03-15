from mrexo import predict_from_measurement
from mrexo.mle_utils import cond_density_quantile

import os
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.stats.mstats import mquantiles

try :
    pwd = os.path.dirname(__file__)
except NameError:
    pwd = ''
    print('Could not find pwd')


'''
Sample script to generate the probability density function for a predicted value.
This is similar to Fig 4 from Kanodia et al. 2019.

'''


measurement_radius = [1,3,10]
r = measurement_radius[1]
result_dir = os.path.join(pwd,'M_dwarfs_dummy')


input_location = os.path.join(result_dir, 'input')
output_location = os.path.join(result_dir, 'output')
weights = np.loadtxt(os.path.join(output_location, 'weights.txt'))
R_points = np.loadtxt(os.path.join(output_location, 'R_points.txt'))
M_points = np.loadtxt(os.path.join(output_location, 'M_points.txt'))
degree = int(np.sqrt(len(weights)))
deg_vec = np.arange(1,degree+1)
qtls = np.linspace(0,1,100)


# If lookup table exists, use lookup table. Else can generate lookup table using generate_lookup_table()
results = predict_from_measurement(measurement = r, measurement_sigma = None,  result_dir = result_dir, qtl=qtls, use_lookup = True)

# PDF is in log scale
predicted_values = np.log10(results[1])
cdf_interp = interp1d(predicted_values, qtls, bounds_error = False, fill_value = 'extrapolate')(M_points)

# Conditional_plot. PDF is derivative of CDF
pdf_interp = np.diff(cdf_interp) / np.diff(M_points)

fig = plt.figure(figsize=(8.5,7))

plt.plot(M_points[:-1], pdf_interp)
plt.ylabel('PDF', fontsize = 20)
plt.xlabel('log Mass ($M_{\oplus}$)', fontsize = 20)
plt.show()
