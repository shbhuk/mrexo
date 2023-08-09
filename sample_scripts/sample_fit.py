import os, sys
from astropy.table import Table
import numpy as np
from multiprocessing import cpu_count
import numpy as np

import pandas as pd

import cProfile
import pstats

import shutil


Platform = sys.platform

if Platform == 'win32':
	HomeDir =  'C:\\Users\\skanodia\\Documents\\\\GitHub\\'
else:
	HomeDir = r"/storage/home/szk381/work/"
	#HomeDir = r"/home/skanodia/work/"


try :
	pwd = os.path.dirname(__file__)
except NameError:
	pwd = os.path.join(HomeDir, 'mrexo', 'sample_scripts')
	print('Could not find pwd')


DataDirectory = os.path.join(HomeDir, 'Mdwarf-Exploration', 'Data', 'MdwarfPlanets')
print(DataDirectory)

t = pd.read_csv(os.path.join(DataDirectory, 'Teff_7000_ExcUpperLimits_20230306RpLt20.csv'))
# t = pd.read_csv(os.path.join(DataDirectory, 'Teff_7000_ExcUpperLimits_20220401_Thesis.csv'))
# t = pd.read_csv(os.path.join(pwd, 'Cool_stars_20181214_exc_upperlim.csv'))
# t = pd.read_csv(os.path.join(pwd, 'Kepler_MR_inputs.csv'))

t = t[~np.isnan(t['pl_insolerr1'])]
t = t[~np.isnan(t['pl_masse'])]


RadiusBounds = [0, 10]# None# [0, 100]
MassBounds = None# [0, 6000]
InsolationBounds = None# [0.01, 5000]
StellarMassBounds = None# [0.2, 1.2]

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
	

RemovePlanets = ['Kepler-54 b', 'Kepler-54 c']
t = t[~np.isin(t.pl_name, RemovePlanets)]


print(len(t))

# In Earth units
Mass = np.array(t['pl_masse'])
# Symmetrical errorbars
MassUSigma = np.array(abs(t['pl_masseerr1']))
MassLSigma = np.array(abs(t['pl_masseerr2']))

Radius = np.array(t['pl_rade'])
# Symmetrical errorbars
RadiusUSigma = np.array(abs(t['pl_radeerr1']))
RadiusLSigma = np.array(abs(t['pl_radeerr2']))

"""
Period = np.array(t['pl_orbper'])
PeriodSigma = np.repeat(np.nan, len(Period))

"""


StellarMass = np.array(t['st_mass'])
StellarMassUSigma = np.array(t['st_masserr1'])
StellarMassLSigma = np.array(t['st_masserr2'])

# Metallicity = np.array(t['st_met'])
# MetallicitySigma = np.array(t['st_meterr1'])

Insolation = np.array(t['pl_insol'])
InsolationUSigma = np.array(t['pl_insolerr1'])
InsolationLSigma = np.array(t['pl_insolerr2'])


Max, Min = 1, 0
Max, Min = np.nan, np.nan


RadiusDict = {'Data': Radius, 'LSigma': RadiusLSigma,  "USigma":RadiusUSigma, 'Max':Max, 'Min':Min, 'Label':'Radius ($R_{\oplus}$)', 'Char':'r'}
MassDict = {'Data': Mass, 'LSigma': MassLSigma, "USigma":MassUSigma,  'Max':Max, 'Min':Min, 'Label':'Mass ($M_{\oplus}$)', 'Char':'m'}

# PeriodDict = {'Data': Period, 'LSigma': PeriodSigma, "USigma":PeriodSigma, 'Max':Max, 'Min':Min, 'Label':'Period (d)', 'Char':'p'}
StellarMassDict = {'Data': StellarMass, 'LSigma': StellarMassLSigma, "USigma":StellarMassUSigma, 'Max':Max, 'Min':Min, 'Label':'Stellar Mass (M$_{\odot}$)', 'Char':'stm'}
#MetallicityDict = {'Data': 10**Metallicity, 'LSigma': np.repeat(np.nan, len(Metallicity)), "USigma":np.repeat(np.nan, len(Metallicity)), 'Max':np.nan, 'Min':np.nan, 'Label':'Metallicity [Fe/H]', 'Char':'feh'}
# MetallicityDict = {'Data': 10**Metallicity, 'LSigma': np.repeat(np.nan, len(Metallicity)), "USigma":np.repeat(np.nan, len(Metallicity)), 'Max':1, 'Min':-0.45, 'Label':'Metallicity [Fe/H]', 'Char':'feh'}
# MetallicityDict = {'Data': 10**(-Metallicity), 'LSigma': np.repeat(np.nan, len(Metallicity)), "USigma":np.repeat(np.nan, len(Metallicity)), 'Max':np.nan, 'Min':np.nan, 'Label':'Metallicity [Fe/H]', 'Char':'feh'}

InsolationDict = {'Data': Insolation, 'LSigma': InsolationLSigma, "USigma":InsolationUSigma, 'Max':Max, 'Min':Min,  'Label':'Pl Insol ($S_{\oplus}$)', 'Char':'insol'}

from mrexo.mle_utils_nd import InputData, MLE_fit
from mrexo.fit_nd import fit_relation
from mrexo.plotting_nd import Plot2DJointDistribution, Plot2DWeights, Plot1DInputDataHistogram
import matplotlib.pyplot as plt

InputDictionaries = [RadiusDict, MassDict, InsolationDict]
# InputDictionaries = [RadiusDict, MassDict, InsolationDict, StellarMassDict]
DataDict = InputData(InputDictionaries)

ndim = len(InputDictionaries)




for d in [20]:#, 40, 80, 100, 500, 1000]:
	# print(d)

	RunName = 'Mdwarf_3D_20220409_M_R_S_bounded'
	RunName = 'Fake_4D_MRSStM'
	RunName = 'GiantPlanet_d60_Bootstrap100_4D_MRSStM'
	RunName = 'AllPlanet_RpLt20_MRS_d60_100MC_100BS'
	RunName = 'AllPlanet_RpLt20_MRS_test'

	# save_path = os.path.join(pwd, 'TestRuns', 'Mdwarf_4D_20220325_M_R_S_StM')
	save_path = os.path.join(pwd, 'TestRuns',  RunName)
	 
	# deg_per_dim = [25, 25, 25, 30]


	# outputs, _ = fit_relation(DataDict, select_deg=34, save_path=save_path, NumBootstrap=0, degree_max=15)

	deg_per_dim = [25, 25, 25]

	# select_deg = 'aic'

	if __name__ == '__main__':

		outputs= fit_relation(DataDict, select_deg=select_deg, save_path=save_path, degree_max=30, cores=2,SymmetricDegreePerDimension=True, NumMonteCarlo=0, NumBootstrap=0)
		#cProfile.run("outputs= fit_relation(DataDict, select_deg=select_deg, save_path=save_path, degree_max=120, cores=40, SymmetricDegreePerDimension=True, NumMonteCarlo=0, NumBootstrap=100)", os.path.join(save_path, 'Profile.prof'))
		shutil.copy(os.path.join(pwd, 'sample_fit.py'), os.path.join(save_path, 'sample_fit_{}.py'.format(RunName)))


		"""
		file = open(os.path.join(save_path, 'FormattedCumulativeProfile.txt'), 'w')
		profile = pstats.Stats(os.path.join(save_path, 'Profile.prof'), stream=file)
		profile.sort_stats('cumulative') # Sorts the result according to the supplied criteria
		profile.print_stats(30) # Prints the first 15 lines of the sorted report
		file.close()

		file = open(os.path.join(save_path, 'FormattedTimeProfile.txt'), 'w')
		profile = pstats.Stats(os.path.join(save_path, 'Profile.prof'), stream=file)
		profile.sort_stats('time') # Sorts the result according to the supplied criteria
		profile.print_stats(30) # Prints the first 15 lines of the sorted report
		file.close()
		"""
		_ = Plot1DInputDataHistogram(save_path)

		if ndim==2:
				
			_ = Plot2DJointDistribution(save_path)
			_ = Plot2DWeights(save_path)

