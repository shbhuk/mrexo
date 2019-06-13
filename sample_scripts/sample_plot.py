import os
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt

from mrexo import plot_r_given_m_relation, plot_m_given_r_relation, plot_mr_and_rm, plot_joint_mr_distribution


try :
    pwd = os.path.dirname(__file__)
except NameError:
    pwd = ''

mdwarf_result = r'C:\Users\shbhu\Documents\Git\mrexo\mrexo\datasets\M_dwarfs_20181214'
mdwarf_result = r'C:\Users\shbhu\Documents\Git\mrexo\mrexo\datasets\M_dwarfs_20181214'
kepler_result = r'C:\Users\shbhu\Documents\Git\mrexo\mrexo\datasets\Kepler_Ning_etal_20170605'

datadir = r"C:/Users/shbhu/Documents/Git/mrexo/use_me"
mdwarf_new_control = os.path.join(datadir,'M_dwarfs_control17')
mdwarf_trappist17 = os.path.join(datadir,'M_dwarfs_wo_Trappist_degree17')
mdwarf_trappistCV = os.path.join(datadir,'M_dwarfs_wo_Trappist_degreeCV')
# kepler_result = r'C:\Users\shbhu\Documents\GitHub\mrexo\mrexo\datasets\Kepler_Ning_etal_20170605'


# result_dir = mdwarf_trappistCV
result_dir = mdwarf_trappistCV

ax = plot_m_given_r_relation(result_dir)
# ax = plot_r_given_m_relation(result_dir)
# ax = plot_mr_and_rm(result_dir)
# ax = plot_joint_mr_distribution(result_dir)

plt.title('M dwarf w/o TRAPPIST-1, 11 degrees', fontsize = 20, pad = 10)

plt.xlim(10**-0.28,19)
plt.ylim(0.02,255)

plt.show()
