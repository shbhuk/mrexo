from mrexo import predict_from_measurement, plot_m_given_r_relation
import os
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt



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

n_posteriors = 1000
random_quantiles = np.random.uniform(0,1, n_posteriors)

result_dir = os.path.join(pwd,'M_dwarfs_dummy')
input_location = os.path.join(result_dir, 'input')
output_location = os.path.join(result_dir, 'output')

R_points = np.loadtxt(os.path.join(output_location, 'R_points.txt'))
M_points = np.loadtxt(os.path.join(output_location, 'M_points.txt'))

# If lookup table exists, use lookup table. Else can generate lookup table using generate_lookup_table()
results = predict_from_measurement(measurement = r, measurement_sigma = None,  result_dir = result_dir, qtl=random_quantiles, use_lookup = True)
# PDF is in log scale
predicted_values = np.log10(results[1])

# cdf_interp = interp1d(predicted_values, qtls, bounds_error = False, fill_value = 'extrapolate')(M_points)


fig, ax1, handles = plot_m_given_r_relation(result_dir)
ax1.plot(np.repeat(np.log10(r), n_posteriors),predicted_values, alpha = 0.6)
# plt.plot()
