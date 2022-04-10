import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import os
from scipy.stats.mstats import mquantiles
from astropy.table import Table

pwd = r"C:\\Users\\shbhu\\Documents\\GitHub\\mrexo\\mrexo"
result_dir = os.path.join(pwd, 'datasets', 'M_dwarfs_20200520')


input_location = os.path.join(result_dir, 'input')
output_location = os.path.join(result_dir, 'output')
aux_output_location = os.path.join(output_location, 'other_data_products')

with open(os.path.join(aux_output_location, 'AxesLabels.txt'), 'r') as f:
	LabelDictionary = eval(f.read())

t = Table.read(os.path.join(input_location, 'XY_inputs.csv'))
Y = t[LabelDictionary['Y_char']]
Y_sigma = t[LabelDictionary['Y_char']+'_sigma']
X = t[LabelDictionary['X_char']]
X_sigma = t[LabelDictionary['X_char']+'_sigma']

Y_min, Y_max = np.loadtxt(os.path.join(input_location, 'Y_bounds.txt'))
X_min, X_max = np.loadtxt(os.path.join(input_location, 'X_bounds.txt'))

X_points = np.loadtxt(os.path.join(output_location, 'X_points.txt'))
Y_cond_X = np.loadtxt(os.path.join(output_location, 'Y_cond_X.txt'))
Y_cond_X_upper = np.loadtxt(os.path.join(output_location, 'Y_cond_X_upper.txt'))
Y_cond_X_lower = np.loadtxt(os.path.join(output_location, 'Y_cond_X_lower.txt'))

weights_boot = np.loadtxt(os.path.join(output_location, 'weights_boot.txt'))
Y_cond_X_boot = np.loadtxt(os.path.join(output_location, 'Y_cond_X_boot.txt'))

n_boot = np.shape(weights_boot)[0]
deg_choose = int(np.sqrt(np.shape(weights_boot[1])))

yx_lower_boot, yx_upper_boot = mquantiles(Y_cond_X_boot, prob=[0.16, 0.84], axis = 0, alphap=1, betap=1).data


fig = plt.figure(figsize=(8.5,7))
plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
ax1 = fig.add_subplot(1,1,1)

ax1.errorbar(x=X, y=Y, xerr=X_sigma, yerr=Y_sigma,fmt='k.',markersize=3, elinewidth=0.3)
ax1.plot(10**X_points, 10**Y_cond_X,  color='maroon', lw=2) # Full dataset run
ax1.fill_between(10**X_points, 10**Y_cond_X_upper, 10**Y_cond_X_lower,alpha=0.3, color='lightsalmon') # Full dataset run
ax1.fill_between(10**X_points, 10**yx_lower_boot, 10**yx_upper_boot,alpha=0.3, color='r') # Bootstrap result

yx_median_line = Line2D([0], [0], color='maroon', lw=2,
	label='Median of f({}$|${}) from full dataset run'.format(LabelDictionary['Y_char'], LabelDictionary['X_char']))
yx_full = mpatches.Patch(color='lightsalmon', alpha=0.3,
	label=r'Quantiles of f({}$|${}) from full dataset run'.format(LabelDictionary['Y_char'], LabelDictionary['X_char']))
yx_boot = mpatches.Patch(color='r', alpha=0.3,
	label=r'Quantiles of the median of the f({}$|${}) from bootstrap'.format(LabelDictionary['Y_char'], LabelDictionary['X_char']))

handles = [yx_median_line, yx_full, yx_boot]


ax1.set_xlabel(LabelDictionary['X_label'])
ax1.set_ylabel(LabelDictionary['Y_label'])
ax1.set_title('f({}$|${}) with degree {}, and {} bootstraps'.format(LabelDictionary['Y_char'], LabelDictionary['X_char'], deg_choose, n_boot), pad=5)
ax1.set_yscale('log')
ax1.set_xscale('log')

# plt.show(block=False)
plt.ylim(10**Y_min, 10**Y_max)
plt.xlim(10**X_min, 10**X_max)
import matplotlib
# matplotlib.rc('text', usetex=True) #use latex for text
plt.legend(handles = handles, prop={'size': 15})
# plt.tight_layout()
