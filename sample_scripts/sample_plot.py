import matplotlib.pyplot as plt
import os
from astropy.table import Table
import numpy as np

from mrexo import plot_mr_relation


try :
    pwd = os.path.dirname(__file__)
except NameError:
    pwd = ''

t = Table.read(os.path.join(pwd,'Cool_stars_20181109.csv'))
result_dir = os.path.join(pwd,'M_dwarfs_deg_11')


t = t.filled()

Mass_sigma = (abs(t['pl_masseerr1']) + abs(t['pl_masseerr2']))/2
Radius_sigma = (abs(t['pl_radeerr1']) + abs(t['pl_radeerr2']))/2

Mass = np.array(t['pl_masse'])
Radius = np.array(t['pl_rade'])

# bounds for Mass and Radius
Radius_min = -0.3
Radius_max = np.log10(max(Radius) + np.std(Radius)/np.sqrt(len(Radius)))
Radius_max = np.log10(max(Radius + Radius_sigma))
Mass_min = np.log10(max(min(Mass) - np.std(Mass)/np.sqrt(len(Mass)), 0.01))
Mass_max = np.log10(max(Mass) + np.std(Mass)/np.sqrt(len(Mass)))

plot_mr_relation(Mass = Mass, Radius = Radius, Mass_sigma = Mass_sigma, Radius_sigma = Radius_sigma, Mass_max = Mass_max,
                Mass_min = Mass_min, Radius_max = Radius_max, Radius_min = Radius_min, result_dir = result_dir)
