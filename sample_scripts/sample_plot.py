import matplotlib.pyplot as plt
import os
from astropy.table import Table
import numpy as np

from mrexo import plot_r_given_m_relation, plot_m_given_r_relation, plot_mr_and_rm


try :
    pwd = os.path.dirname(__file__)
except NameError:
    pwd = ''

result_dir = os.path.join(pwd,'M_dwarfs_deg15')
result_dir = "C:/Users/shbhu/Box Sync/M_dwarves/straight_line_simulation/simulation_50_points"

#result_dir = "C:/Users/shbhu/Documents/GitHub/mrexo/sample_kepler2/Kepler_55_open_corrected"
#result_dir = "C:/Users/shbhu/Documents/GitHub/mrexo/sample_kepler2/Kepler_55_cluster"
#ax, handles = plot_r_given_m_relation(result_dir = result_dir)
ax, handles = plot_mr_and_rm(result_dir=result_dir)


