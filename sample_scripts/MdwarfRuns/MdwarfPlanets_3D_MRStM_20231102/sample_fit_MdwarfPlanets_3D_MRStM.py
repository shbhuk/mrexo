import os, sys
from astropy.table import Table
import numpy as np
import numpy as np
import pandas as pd
import shutil

from mrexo.mle_utils_nd import InputData, MLE_fit
from mrexo.fit_nd import fit_relation
from mrexo.plotting_nd import Plot2DJointDistribution, Plot2DWeights, Plot1DInputDataHistogram
import matplotlib.pyplot as plt

Platform = sys.platform

if Platform == 'win32':
	HomeDir =  'C:\\Users\\skanodia\\Documents\\\\GitHub\\'
else:
	HomeDir = r"/storage/home/szk381/work/"
	HomeDir = r"/home/skanodia/work/"


try :
	pwd = os.path.dirname(__file__)
except NameError:
	pwd = os.path.join(HomeDir, 'mrexo', 'sample_scripts')
	print('Could not find pwd')

# Directory with dataset to be fit
DataDirectory = os.path.join(HomeDir, 'Mdwarf-Exploration', 'Data', 'MdwarfPlanets')
print(DataDirectory)

t = pd.read_csv(os.path.join(DataDirectory, 'Teff_4200_ExcUpperLimits_20231102.csv'))

# Mask NaNs
t = t[~np.isnan(t['pl_insolerr1'])]
t = t[~np.isnan(t['pl_masse'])]

# Define bounds in different dimensions
RadiusBounds = [0, 20]# None# [0, 100]
MassBounds = None# [0, 6000]
InsolationBounds = None# [0.01, 5000]
StellarMassBounds = None# [0.2, 1.2]

# Measurements with very small errors are set to NaN to avoid integration errors
t['st_masserr1'][t['st_masserr1'] < 0.005] = np.nan
t['st_masserr2'][t['st_masserr2'] < 0.005] = np.nan

if RadiusBounds is not None:
	t = t[(t.pl_rade > RadiusBounds[0]) & (t.pl_rade < RadiusBounds[1])]

if MassBounds is not None:
	t = t[(t.pl_masse > MassBounds[0]) & (t.pl_masse < MassBounds[1])]

if InsolationBounds is not None:
	t = t[(t.pl_insol > InsolationBounds[0]) & (t.pl_insol < InsolationBounds[1])]
	
if StellarMassBounds is not None:
	t = t[(t.st_mass > StellarMassBounds[0]) & (t.st_mass < StellarMassBounds[1])]
	
# Remove particular planets
RemovePlanets = ['Kepler-54 b', 'Kepler-54 c']
t = t[~np.isin(t.pl_name, RemovePlanets)]

print(len(t))

# In Earth units
Mass = np.array(t['pl_masse'])
# Asymmetrical errorbars
MassUSigma = np.array(abs(t['pl_masseerr1']))
MassLSigma = np.array(abs(t['pl_masseerr2']))

Radius = np.array(t['pl_rade'])
# Asymmetrical errorbars
RadiusUSigma = np.array(abs(t['pl_radeerr1']))
RadiusLSigma = np.array(abs(t['pl_radeerr2']))

StellarMass = np.array(t['st_mass'])
StellarMassUSigma = np.array(t['st_masserr1'])
StellarMassLSigma = np.array(t['st_masserr2'])

Insolation = np.array(t['pl_insol'])
InsolationUSigma = np.array(t['pl_insolerr1'])
InsolationLSigma = np.array(t['pl_insolerr2'])

# Let the script pick the max and min bounds, or can hard code those in. Note that the dataset must fall within the bounds if they are specified.
Max, Min = np.nan, np.nan

# Define input dictionary for each dimension
RadiusDict = {'Data': Radius, 'LSigma': RadiusLSigma,  "USigma":RadiusUSigma, 'Max':Max, 'Min':Min, 'Label':'Radius ($R_{\oplus}$)', 'Char':'r'}
MassDict = {'Data': Mass, 'LSigma': MassLSigma, "USigma":MassUSigma,  'Max':Max, 'Min':Min, 'Label':'Mass ($M_{\oplus}$)', 'Char':'m'}
# PeriodDict = {'Data': Period, 'LSigma': PeriodSigma, "USigma":PeriodSigma, 'Max':Max, 'Min':Min, 'Label':'Period (d)', 'Char':'p'}
StellarMassDict = {'Data': StellarMass, 'LSigma': StellarMassLSigma, "USigma":StellarMassUSigma, 'Max':Max, 'Min':Min, 'Label':'Stellar Mass (M$_{\odot}$)', 'Char':'stm'}
InsolationDict = {'Data': Insolation, 'LSigma': InsolationLSigma, "USigma":InsolationUSigma, 'Max':Max, 'Min':Min,  'Label':'Pl Insol ($S_{\oplus}$)', 'Char':'insol'}

# 3D fit with planetary radius, mass and insolation
InputDictionaries = [RadiusDict, MassDict, StellarMassDict]
DataDict = InputData(InputDictionaries)
ndim = len(InputDictionaries)

RunName = 'MdwarfPlanets_3D_MRStM'
save_path = os.path.join(pwd, 'TestRuns',  RunName)

# Harcode the  number of degrees per dimension. Read the `fit_relation()` documentation for alternatives
deg_per_dim = [25, 25, 25]

if __name__ == '__main__':

	outputs= fit_relation(DataDict, select_deg='cv', save_path=save_path, degree_max=60, cores=20,SymmetricDegreePerDimension=True, NumMonteCarlo=0, NumBootstrap=0)
	shutil.copy(os.path.join(pwd, 'sample_fit.py'), os.path.join(save_path, 'sample_fit_{}.py'.format(RunName)))

	_ = Plot1DInputDataHistogram(save_path)

	if ndim==2:
			
		_ = Plot2DJointDistribution(save_path)
		_ = Plot2DWeights(save_path)

