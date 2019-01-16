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



sim_sizes = [20,50,100]
#sim_sizes = [10]
intrinsic_disp = [0.0,0.1,0.5]
#intrinsic_disp = [1.0]

plt.close()
#
# for i in sim_sizes:
#     for j in intrinsic_disp:
#         print(i,j)
#         result_dir = os.path.join(directory, 'Simulation_{}pts_{}disp'.format(i, j))
#
#         fig, ax, handles = plot_mr_and_rm(result_dir = result_dir)
#         ax.set_title('{} points, {}*log M dispersion'.format(i,j))
#         # ax.legend().set_visible(False)
#         break
#     break
#
#
#         # plt.savefig(os.path.join(pwd, 'Sim_{}pts_{}disp.png'.format(i, j)))
#         # plt.close()
#
#         ax = plot_joint_mr_distribution(result_dir, include_conditionals = False)
#         ax.set_title('{} points, {}*log M dispersion'.format(i,j))
#         plt.savefig(os.path.join(pwd, 'Sim_{}pts_{}disp_jointdist.png'.format(i, j)))
#         # plt.close()







for i in range(0,9):

    # result_dir =  "C:/Users/shbhu/Documents/Git/mrexo/straight_line_simulation/Simulation_{}_points".format(20)
    #result_dir =  "C:/Users/shbhu/Documents/GitHub/mrexo/sample_scripts/M_dwarfs_deg_increase_bounds2{}".format(17)
    #result_dir =  "C:/Users/shbhu/Documents/GitHub/mrexo/sample_scripts/M_dwarfs_deg_cancel_boundary_poly17"
    result_dir = "C:/Users/shbhu/Documents/Git/mrexo/sample_scripts/M_dwarfs_deg17_final"
    result_dir = r"C:\Users\shbhu\Documents\Git\mrexo\mrexo\datasets\Kepler_Ning_etal_20170605"
    # result_dir = r"C:\Users\shbhu\Documents\Git\mrexo\sample_kepler2\Kepler_wo_boundary"
    # fig, ax, handles = plot_m_given_r_relation(result_dir = result_dir)
    # fig, ax, handles = plot_r_given_m_relation(result_dir = result_dir)
    fig, ax, handles = plot_mr_and_rm(result_dir=result_dir)
    ax.set_title('Kepler conditional distributions')
    plt.xlim(-0.3, 1.30483)
    plt.ylim(-1.744727, 2.44790)

    _ = plot_mr_and_rm(result_dir)
    ax = plot_joint_mr_distribution(result_dir, include_conditionals = False)
    ax.set_title('Kepler joint distributions')
    plt.xlim(-0.3, 1.30483)
    plt.ylim(-1.744727, 2.44790)
    plt.show()
    break



weights = np.loadtxt(r'C:/Users/shbhu/Documents/Git/mrexo/mrexo/datasets/M_dwarfs_20181214/output/weights.txt')
plt.imshow(np.reshape(weights, [17,17]))
