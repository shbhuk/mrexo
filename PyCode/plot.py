import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
import os,sys
from scipy.stats.mstats import mquantiles
pwd = os.path.dirname(__file__)
#sys.path.append(pwd)
import MLE_fit
import importlib
importlib.reload(MLE_fit)

print(pwd)

result_dir = os.path.join(pwd,'Bootstrap_results_cluster50')
result_dir = os.path.join(pwd,'Bootstrap_cyberlamp_parallel_full2')
result_dir = os.path.join(pwd,'Results','M_dwarfs_logtrue')
result_dir = os.path.join(pwd,'Mdwarf_CV_100boot')
result_dir = os.path.join(pwd,'test')

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
ax1.set_title('Mdwarf data: Mass - radius relations with degree {}'.format(deg_choose))

plt.show()

####################
'''
M_1 = np.log10(1)
M_10 = np.log10(10)
M_50 = np.log10(50)
M_100 = np.log10(100)

R_seq = np.linspace(Radius_min,Radius_max, 100)
density_M_1 = np.zeros((100,n_boot))
density_M_10 = np.zeros((100,n_boot))
density_M_50 = np.zeros((100,n_boot))
density_M_100 = np.zeros((100,n_boot))

for i in range(0,n_boot):
    weights_t = np.transpose(np.reshape(weights_boot[i,:],newshape = [deg_choose,deg_choose])).flatten()
    print(i)
    for j in range(0,len(R_seq)):
        density_M_1[j,i] = sum(MLE_fit.conditional_density(y = M_1, y_max = Mass_max, y_min = Mass_min, x = R_seq[j], x_max = Radius_max, x_min = Radius_min, deg = deg_choose, w_hat = weights_t))
        density_M_10[j,i] = sum(MLE_fit.conditional_density(y = M_10, y_max = Mass_max, y_min = Mass_min, x = R_seq[j], x_max = Radius_max, x_min = Radius_min, deg = deg_choose, w_hat = weights_t))
        density_M_50[j,i] = sum(MLE_fit.conditional_density(y = M_50, y_max = Mass_max, y_min = Mass_min, x = R_seq[j], x_max = Radius_max, x_min = Radius_min, deg = deg_choose, w_hat = weights_t))
        density_M_100[j,i] = sum(MLE_fit.conditional_density(y = M_100, y_max = Mass_max, y_min = Mass_min, x = R_seq[j], x_max = Radius_max, x_min = Radius_min, deg = deg_choose, w_hat = weights_t))

density_M_1_quantile = mquantiles(density_M_1,prob = [0.16, 0.5, 0.84],axis = 1,alphap=1,betap=1).data
density_M_10_quantile = mquantiles(density_M_10,prob = [0.16, 0.5, 0.84],axis = 0,alphap=1,betap=1).data
density_M_50_quantile = mquantiles(density_M_50,prob = [0.16, 0.5, 0.84],axis = 0,alphap=1,betap=1).data
density_M_100_quantile = mquantiles(density_M_100,prob = [0.16, 0.5, 0.84],axis = 0,alphap=1,betap=1).data


fig = plt.figure()
ax2 = fig.add_subplot(1,1,1)

M_1_ind = np.where((R_seq > -0.4) & (R_seq < 1.2))[0]
M_10_ind = np.where((R_seq > -0.3) & (R_seq < 1.2))[0]

lower = density_M_1_quantile[M_1_ind][:,0]
upper = density_M_1_quantile[M_1_ind][:,2]

ax2.plot(10**R_seq[M_1_ind],density_M_1_quantile[M_1_ind][:,1])
ax2.fill_between(10**R_seq[M_1_ind],lower,upper,alpha = 0.5)
ax2.plot(R_seq[M_10_ind],density_M_10_quantile[M_10_ind])
ax2.fill_between(R_points,lower_boot,upper_boot,alpha = 0.5)

'''
