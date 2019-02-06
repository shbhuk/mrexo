import os
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt

from mrexo import plot_r_given_m_relation, plot_m_given_r_relation, plot_mr_and_rm, plot_joint_mr_distribution


try :
    pwd = os.path.dirname(__file__)
except NameError:
    pwd = ''

result_dir = os.path.join(pwd,'M_dwarfs_deg15')
result_dir = "C:/Users/shbhu/Box Sync/M_dwarves/straight_line_simulation/simulation_50_points"

result_dir = "C:/Users/shbhu/Documents/Git/mrexo/sample_kepler2/Kepler_55_open_corrected"
directory = "C:/Users/shbhu/Documents/Git/mrexo/straight_line_simulation"


mdwarf_result = r'C:\Users\shbhu\Documents\Git\mrexo\mrexo\datasets\M_dwarfs_20181214'
kepler_result = r'C:\Users\shbhu\Documents\Git\mrexo\mrexo\datasets\Kepler_Ning_etal_20170605'


result_dir = mdwarf_result

ax = plot_m_given_r_relation(result_dir)
plt.title('M dwarf conditional distributions', fontsize = 20, pad = 2)
plt.xlim(-0.25, 1.25483)
plt.ylim(-1.744727, 2.44790)

plt.yticks(np.arange(-1.5, 2.5, 0.5))
plt.xticks(np.arange(-0.25, 1.5, 0.25))
