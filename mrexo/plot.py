import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
import os,sys
from scipy.stats.mstats import mquantiles
pwd = os.path.dirname(__file__)

print(pwd)

result_dir = os.path.join(pwd,'Bootstrap_results_cluster50')
result_dir = os.path.join(pwd,'Bootstrap_cyberlamp_parallel_full2')
#result_dir = os.path.join(pwd,'Results','M_dwarfs_logtrue')
result_dir = os.path.join(pwd,'M_dwarfs_degree_11')
#result_dir = os.path.join(pwd,'Full_run_CV1')
#result_dir = os.path.join(pwd,'test')

t = Table.read(os.path.join(pwd,'MR_Kepler_170605_noanalytTTV_noupplim.csv'))
t = Table.read(os.path.join(pwd,'Cool_stars_20181109.csv'))

t = t.filled()





M_sigma = (abs(t['pl_masseerr1']) + abs(t['pl_masseerr2']))/2
R_sigma = (abs(t['pl_radeerr1']) + abs(t['pl_radeerr2']))/2

M_obs = np.array(t['pl_masse'])
R_obs = np.array(t['pl_rade'])



# bounds for Mass and Radius
Radius_min = -0.3
Radius_max = np.log10(max(R_obs) + np.std(R_obs)/np.sqrt(len(R_obs)))
Mass_min = np.log10(max(min(M_obs) - np.std(M_obs)/np.sqrt(len(M_obs)), 0.1))
Mass_max = np.log10(max(M_obs) + np.std(M_obs)/np.sqrt(len(M_obs)))


def plot_mr_relation(Mass_sig):



M_points = np.loadtxt(os.path.join(result_dir, 'M_points.txt'))
R_points = np.loadtxt(os.path.join(result_dir, 'R_points.txt'))
weights = np.loadtxt(os.path.join(result_dir, 'weights.txt'))
Mass_marg = np.loadtxt(os.path.join(result_dir, 'Mass_marg.txt'))
Radius_marg = np.loadtxt(os.path.join(result_dir, 'Radius_marg.txt'))
M_cond_R = np.loadtxt(os.path.join(result_dir, 'M_cond_R.txt'))
R_cond_M = np.loadtxt(os.path.join(result_dir, 'R_cond_M.txt'))
M_cond_R_var = np.loadtxt(os.path.join(result_dir, 'M_cond_R_var.txt'))
M_cond_R_upper = np.loadtxt(os.path.join(result_dir, 'M_cond_R_upper.txt'))
M_cond_R_lower = np.loadtxt(os.path.join(result_dir, 'M_cond_R_lower.txt'))
R_cond_M_var = np.loadtxt(os.path.join(result_dir, 'R_cond_M_var.txt'))
R_cond_M_upper = np.loadtxt(os.path.join(result_dir, 'R_cond_M_upper.txt'))
R_cond_M_lower = np.loadtxt(os.path.join(result_dir, 'R_cond_M_lower.txt'))


weights_boot = np.loadtxt(os.path.join(result_dir, 'weights_boot.txt'))
Mass_marg_boot = np.loadtxt(os.path.join(result_dir, 'Mass_marg_boot.txt'))
Radius_marg_boot = np.loadtxt(os.path.join(result_dir, 'Radius_marg_boot.txt'))
M_cond_R_boot = np.loadtxt(os.path.join(result_dir, 'M_cond_R_boot.txt'))
R_cond_M_boot = np.loadtxt(os.path.join(result_dir, 'R_cond_M_boot.txt'))
M_cond_R_var_boot = np.loadtxt(os.path.join(result_dir, 'M_cond_R_var_boot.txt'))
M_cond_R_upper_boot = np.loadtxt(os.path.join(result_dir, 'M_cond_R_upper_boot.txt'))
M_cond_R_lower_boot = np.loadtxt(os.path.join(result_dir, 'M_cond_R_lower_boot.txt'))
R_cond_M_var_boot = np.loadtxt(os.path.join(result_dir, 'R_cond_M_var_boot.txt'))
R_cond_M_upper_boot = np.loadtxt(os.path.join(result_dir, 'R_cond_M_upper_boot.txt'))
R_cond_M_lower_boot = np.loadtxt(os.path.join(result_dir, 'R_cond_M_lower_boot.txt'))

n_boot = np.shape(weights_boot)[0]
deg_choose = int(np.sqrt(np.shape(weights_boot[1])))
#deg_choose = 5

logMass = np.log10(M_obs)
logRadius = np.log10(R_obs)


#logMass = np.log10(M_points[0])
#logRadius = np.log10(R_points[0])

logMass_sigma = 0.434 * M_sigma/M_obs
logRadius_sigma = 0.434 * R_sigma/R_obs

lower_boot, upper_boot = mquantiles(M_cond_R_boot,prob = [0.16, 0.84],axis = 0,alphap=1,betap=1).data
#lower_boot, upper_boot = np.mean(M_cond_R_lower_boot, axis = 0), np.mean(M_cond_R_upper_boot, axis = 0)


fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

ax1.errorbar(x = logRadius, y = logMass, xerr = logRadius_sigma, yerr = logMass_sigma,fmt = 'k.',markersize = 2, elinewidth = 0.3)
#ax1.errorbar(x = R_obs, y = M_obs, xerr = R_sigma/R_obs, yerr = M_sigma/M_obs,fmt = 'k.',markersize = 2, elinewidth = 0.3)
ax1.plot(R_points,M_cond_R) # Non parametric result
ax1.fill_between(R_points,M_cond_R_lower,M_cond_R_upper,alpha = 0.3, color = 'r') # Non parametric result
ax1.fill_between(R_points,lower_boot,upper_boot,alpha = 0.5) # Bootstrap result


ax1.set_xlabel('log Radius (Earth Radii)')
ax1.set_ylabel('log Mass (Earth Mass)')
#ax1.set_title('Mdwarf data: Mass - radius relations with degree {}'.format(deg_choose))
ax1.set_title('Kepler data: Mass - radius relations with degree {}'.format(deg_choose))

plt.show()
