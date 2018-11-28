import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import os
from scipy.stats.mstats import mquantiles
from astropy.table import Table


def plot_mr_relation(result_dir):


    input_location = os.path.join(result_dir, 'input')
    output_location = os.path.join(result_dir, 'output')



    
    t = Table.read(os.path.join(input_location, 'MR_inputs.csv'))
    Mass = t['pl_masse']
    Mass_sigma = t['pl_masseerr1']
    Radius = t['pl_rade']
    Radius_sigma = t['pl_radeerr1']

    Mass_min, Mass_max = np.loadtxt(os.path.join(input_location, 'Mass_bounds.txt'))
    Radius_min, Radius_max = np.loadtxt(os.path.join(input_location, 'Radius_bounds.txt'))

    R_points = np.loadtxt(os.path.join(output_location, 'R_points.txt'))
    M_cond_R = np.loadtxt(os.path.join(output_location, 'M_cond_R.txt'))
    M_cond_R_upper = np.loadtxt(os.path.join(output_location, 'M_cond_R_upper.txt'))
    M_cond_R_lower = np.loadtxt(os.path.join(output_location, 'M_cond_R_lower.txt'))

    weights_boot = np.loadtxt(os.path.join(output_location, 'weights_boot.txt'))
    M_cond_R_boot = np.loadtxt(os.path.join(output_location, 'M_cond_R_boot.txt'))

    n_boot = np.shape(weights_boot)[0]
    deg_choose = int(np.sqrt(np.shape(weights_boot[1])))

    logMass = np.log10(Mass)
    logRadius = np.log10(Radius)

    logMass_sigma = 0.434 * Mass_sigma/Mass
    logRadius_sigma = 0.434 * Radius_sigma/Radius

    lower_boot, upper_boot = mquantiles(M_cond_R_boot,prob = [0.16, 0.84],axis = 0,alphap=1,betap=1).data

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)

    ax1.errorbar(x = logRadius, y = logMass, xerr = logRadius_sigma, yerr = logMass_sigma,fmt = 'k.',markersize = 2, elinewidth = 0.3)
    ax1.plot(R_points,M_cond_R,  color = 'darkblue', lw = 2) # Full dataset run
    ax1.fill_between(R_points,M_cond_R_lower,M_cond_R_upper,alpha = 0.3, color = 'r') # Full dataset run
    ax1.fill_between(R_points,lower_boot,upper_boot,alpha = 0.3, color = 'b') # Bootstrap result



    mean_line = Line2D([0], [0], color='darkblue', lw = 2,label = 'Mean of conditional distribution of M given R from full dataset run')
    red_patch = mpatches.Patch(color='r', alpha = 0.3,  label=r'16%-84% quantile of conditional distribution of M given R from full dataset run  ')
    blue_patch = mpatches.Patch(color='b', alpha = 0.3, label=r'16%-84% quantile of the MEAN of the conditional distribution of M given R from bootstrap')

    plt.legend(handles=[mean_line, red_patch, blue_patch])


    ax1.set_xlabel('log Radius (Earth Radii)')
    ax1.set_ylabel('log Mass (Earth Mass)')
    ax1.set_title('Mass - radius relations with degree {}'.format(deg_choose))

    plt.show()
