import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
import os
from scipy.stats.mstats import mquantiles



def plot_mr_relation(M_obs, Mass_sigma, Mass_min, Mass_max, R_obs, Radius_sigma, Radius_min, Radius_max, result_dir):




    R_points = np.loadtxt(os.path.join(result_dir, 'R_points.txt'))
    M_cond_R = np.loadtxt(os.path.join(result_dir, 'M_cond_R.txt'))
    M_cond_R_upper = np.loadtxt(os.path.join(result_dir, 'M_cond_R_upper.txt'))
    M_cond_R_lower = np.loadtxt(os.path.join(result_dir, 'M_cond_R_lower.txt'))

    weights_boot = np.loadtxt(os.path.join(result_dir, 'weights_boot.txt'))
    M_cond_R_boot = np.loadtxt(os.path.join(result_dir, 'M_cond_R_boot.txt'))


    n_boot = np.shape(weights_boot)[0]
    deg_choose = int(np.sqrt(np.shape(weights_boot[1])))

    logMass = np.log10(M_obs)
    logRadius = np.log10(R_obs)

    logMass_sigma = 0.434 * Mass_sigma/M_obs
    logRadius_sigma = 0.434 * Radius_sigma/R_obs

    lower_boot, upper_boot = mquantiles(M_cond_R_boot,prob = [0.16, 0.84],axis = 0,alphap=1,betap=1).data

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)

    ax1.errorbar(x = logRadius, y = logMass, xerr = logRadius_sigma, yerr = logMass_sigma,fmt = 'k.',markersize = 2, elinewidth = 0.3)
    ax1.plot(R_points,M_cond_R) # Non parametric result
    ax1.fill_between(R_points,M_cond_R_lower,M_cond_R_upper,alpha = 0.3, color = 'r') # Non parametric result
    ax1.fill_between(R_points,lower_boot,upper_boot,alpha = 0.5) # Bootstrap result


    ax1.set_xlabel('log Radius (Earth Radii)')
    ax1.set_ylabel('log Mass (Earth Mass)')
    #ax1.set_title('Mdwarf data: Mass - radius relations with degree {}'.format(deg_choose))
    #ax1.set_title('Kepler data: Mass - radius relations with degree {}'.format(deg_choose))

    plt.show()
