import matplotlib.pyplot as plt
import os
from astropy.table import Table
import numpy as np

from mrexo import plot_r_given_m_relation, plot_m_given_r_relation


try :
    pwd = os.path.dirname(__file__)
except NameError:
    pwd = ''

result_dir = os.path.join(pwd,'M_dwarfs_cv5')

#result_dir = "C:/Users/shbhu/Documents/Git/mrexo/sample_kepler2/Kepler_55_open_corrected"

ax, handles = plot_m_given_r_relation(result_dir = result_dir)
#ax = plot_r_given_m_relation(result_dir = result_dir)


from matplotlib.lines import Line2D

ax.errorbar(x = 0, y = 0, yerr = 1, xerr = 1,fmt = 'r.',markersize = 3, elinewidth = 0.5)
handles.append(Line2D([0], [0], color='r', marker='s',  label='Predicted value'))

plt.legend(handles = handles)

'''
ax.errorbar(x = logRadius, y = predicted_mean, xerr = Radius_sigma, yerr = [predicted_lower_quantile,predicted_upper_quantile],fmt = 'r.',markersize = 3, elinewidth = 0.5, label =' Predicted')
plt.hlines(predicted_mean, linestyles = 'dashed')
plt.vlines(logRadius, linestyles = 'dashed')

plt.legend(handles = handles)
'''
