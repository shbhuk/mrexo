import os
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt

# from mrexo import plot_r_given_m_relation, plot_m_given_r_relation, plot_mr_and_rm, plot_joint_mr_distribution
from mrexo.plot_beta import plot_x_given_y_relation, plot_y_given_x_relation, plot_yx_and_xy, plot_joint_xy_distribution


try :
    pwd = os.path.dirname(__file__)
except NameError:
    pwd = ''

mdwarf_result = r'C:\Users\shbhu\Documents\Git\mrexo\mrexo\datasets\M_dwarfs_20200116'
kepler_result = r'C:\Users\shbhu\Documents\Git\mrexo\mrexo\datasets\Kepler_Ning_etal_20170605'

# FP_result = r'C:\\Users\\shbhu\\Box Sync\\M_dwarves\\MassRadiusPeriod\\FP2018_RadPer_SampleSize50_Trial1'
# FP_result = r'C:\\Users\\shbhu\\Box Sync\\M_dwarves\\MassRadiusPeriod\\PR_50Trial0_deg15'

new_cv = os.path.join(pwd,'Mdwarfs_20200421')


result_dir = new_cv

# # Plot the conditional distribution f(m|r)
ax = plot_y_given_x_relation(result_dir)



# ax[1].set_xlabel('Radius')
# ax[1].set_ylabel('Period')
#
# # Plot the conditional distribution f(r|m)
ax = plot_x_given_y_relation(result_dir)

#
# # Plot both the conditional distributions f(m|r) and f(r|m), similar to Kanodia 2019, Fig 3.
ax = plot_yx_and_xy(result_dir)

# ax = plot_mr_and_rm(result_dir)
# ax[1].set_xlabel('Radius')
# ax[1].set_ylabel('Period')

# Plot the joint distribution f(m,r)
ax = plot_joint_xy_distribution(result_dir)

# ax[1].set_xlabel('Radius')
# ax[1].set_ylabel('Period')

plt.show()
