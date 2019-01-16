import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import os
from scipy.stats.mstats import mquantiles
from scipy.interpolate import interp1d
from astropy.table import Table

import sys

from mrexo.predict import find_mass_probability_distribution_function

query_radius = [1]


result_dir = "C:/Users/shbhu/Documents/Git/mrexo/sample_scripts/M_dwarfs_deg17_final"

input_location = os.path.join(result_dir, 'input')
output_location = os.path.join(result_dir, 'output')

R_points = np.loadtxt(os.path.join(output_location, 'R_points.txt'))
M_points = np.loadtxt(os.path.join(output_location, 'M_points.txt'))

weights_mle = np.loadtxt(os.path.join(output_location,'weights.txt'))
weights_boot = np.loadtxt(os.path.join(output_location,'weights_boot.txt'))
degree = int(np.sqrt(len(weights_mle)))
deg_vec = np.arange(1,degree+1)

Mass_min, Mass_max = np.loadtxt(os.path.join(input_location, 'Mass_bounds.txt'))
Radius_min, Radius_max = np.loadtxt(os.path.join(input_location, 'Radius_bounds.txt'))

for r in query_radius:
    pdf_interp, lower_boot, upper_boot = find_mass_probability_distribution_function(r, Radius_min, Radius_max, Mass_max, Mass_min, weights_mle, weights_boot, degree, deg_vec, M_points)

    plt.plot(M_points[:-1], pdf_interp)
    plt.fill_between(M_points[:-1], lower_boot,upper_boot,alpha=0.3)
    plt.text(np.median(pdf_interp),0.5 'Radius = {}'.format(r), size = 20)

plt.ylim(0,1)
plt.show()
