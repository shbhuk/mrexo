import matplotlib.pyplot as plt
import os
from astropy.table import Table
import numpy as np

from mrexo import plot_mr_relation


try :
    pwd = os.path.dirname(__file__)
except NameError:
    pwd = ''

result_dir = os.path.join(pwd,'M_dwarfs_11')

plot_mr_relation(result_dir = result_dir)
