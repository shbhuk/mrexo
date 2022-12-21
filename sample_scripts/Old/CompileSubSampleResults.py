import os, sys
#from astropy.table import Table
import numpy as np
import matplotlib
import numpy as np
import glob

import pandas as pd

Platform = sys.platform

if Platform == 'win32':
	HomeDir =  'C:\\Users\\shbhu\\Documents\\\\GitHub\\'
else:
	HomeDir = r"/storage/home/szk381/work/"


try :
	pwd = os.path.dirname(__file__)
except NameError:
	pwd = os.path.join(HomeDir, 'mrexo', 'sample_scripts')
	print('Could not find pwd')
 
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt 
 
#RunName = "Kepler_HFR2020b_RP_deg500_0_4_SubSample_1000_Trial_*" 
RunName = "Kepler_HFR2020b_RP_deg600_0_4_SubSample_1000_Trial_*"
save_path = os.path.join(pwd, 'TestRuns', RunName) 
ListofRuns = glob.glob(save_path)

NDegreesAll = np.zeros((len(ListofRuns), 2))



for i, run in enumerate(ListofRuns):
  RunPath = ListofRuns[i]
  OutputDir = os.path.join(RunPath, 'output')
  
  try:
    NDegrees = np.loadtxt(os.path.join(OutputDir, 'deg_per_dim.txt'))
    NDegreesAll[i] = NDegrees
  except:
    print(i)
  print(ListofRuns[i], NDegrees)
  
plt.scatter(NDegreesAll[:,0], NDegreesAll[:, 1])
plt.show()
