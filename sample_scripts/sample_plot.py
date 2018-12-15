import os
from astropy.table import Table
import numpy as np

from mrexo import plot_r_given_m_relation, plot_m_given_r_relation, plot_mr_and_rm, plot_joint_mr_distribution


try :
    pwd = os.path.dirname(__file__)
except NameError:
    pwd = ''

result_dir = os.path.join(pwd,'M_dwarfs_deg15')
result_dir = "C:/Users/shbhu/Box Sync/M_dwarves/straight_line_simulation/simulation_50_points"

result_dir = "C:/Users/shbhu/Documents/Git/mrexo/sample_kepler2/Kepler_55_open_corrected"
result_dir = "C:/Users/shbhu/Documents/Git/mrexo/straight_line_simulation"



for i in range(10,20):
    i = 17
    result_dir =  "C:/Users/shbhu/Documents/Git/mrexo/straight_line_simulation/simulation_{}_points".format(20)
    result_dir =  "C:/Users/shbhu/Documents/GitHub/mrexo/sample_scripts/M_dwarfs_deg_increase_bounds2{}".format(17)
    result_dir =  "C:/Users/shbhu/Documents/GitHub/mrexo/sample_scripts/M_dwarfs_deg_cancel_boundary_poly17"
    fig, ax, handles = plot_m_given_r_relation(result_dir = result_dir)
    # ax, handles = plot_m_given_r_relation(result_dir = result_dir)
    #ax, handles = plot_mr_and_rm(result_dir=result_dir)


    ax = plot_joint_mr_distribution(result_dir, include_conditionals = False)
    break
