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



sim_sizes = [20,50,100, 200]
sim_sizes = [10]
intrinsic_disp = [0, 0.1,0.5,1]
intrinsic_disp = [1.0]



for i in sim_sizes:
    for j in intrinsic_disp:

        result_dir = os.path.join(directory, 'Simulation_{}pts_{}disp'.format(i, j))

        fig, ax, handles = plot_m_given_r_relation(result_dir = result_dir)
        ax.set_title('{} points ,{}*log M dispersion'.format(i,j))

        plt.savefig(os.path.join(pwd, 'Sim_{}pts_{}disp.png'.format(i, j)))

        ax = plot_joint_mr_distribution(result_dir, include_conditionals = False)
        ax.set_title('{} points ,{}*log M dispersion'.format(i,j))
        #plt.savefig(os.path.join(pwd, 'Sim_{}pts_{}disp_jointdist.png'.format(i, j)))






'''
for i in range(10,20):
    i = 17
    result_dir =  "C:/Users/shbhu/Documents/Git/mrexo/straight_line_simulation/Simulation_{}_points".format(20)
    #result_dir =  "C:/Users/shbhu/Documents/GitHub/mrexo/sample_scripts/M_dwarfs_deg_increase_bounds2{}".format(17)
    #result_dir =  "C:/Users/shbhu/Documents/GitHub/mrexo/sample_scripts/M_dwarfs_deg_cancel_boundary_poly17"
    #result_dir = "C:/Users/shbhu/Documents/Git/mrexo/sample_scripts/M_dwarfs_deg17_trimmed"
    fig, ax, handles = plot_m_given_r_relation(result_dir = result_dir)
    # ax, handles = plot_m_given_r_relation(result_dir = result_dir)
    #ax, handles = plot_mr_and_rm(result_dir=result_dir)


    ax = plot_joint_mr_distribution(result_dir, include_conditionals = False)
    break
'''
