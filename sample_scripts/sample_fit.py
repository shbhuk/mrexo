import os, sys
from astropy.table import Table
import numpy as np
from multiprocessing import cpu_count
import numpy as np

import pandas as pd

import cProfile
import pstats


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


DataDirectory = os.path.join(HomeDir, 'Mdwarf-Exploration', 'Data', 'MdwarfPlanets')
print(DataDirectory)

UTeff = 7000

# t = pd.read_csv(os.path.join(DataDirectory, 'Teff_{}_ExcUpperLimits_20210823.csv'.format(UTeff)))
t = pd.read_csv(os.path.join(DataDirectory, 'Teff_7000_ExcUpperLimits_20230214_RpGt20.csv'))
# t = pd.read_csv(os.path.join(DataDirectory, 'Teff_7000_ExcUpperLimits_20220401_Thesis.csv'))
# t = pd.read_csv(os.path.join(pwd, 'Cool_stars_20181214_exc_upperlim.csv'))
# t = pd.read_csv(os.path.join(pwd, 'Kepler_MR_inputs.csv'))

t = t[~np.isnan(t['pl_insolerr1'])]

RadiusBounds = [5, 12]
MassBounds = [5, 3000]
InsolationBounds = None# [0.01, 5000]
StellarMassBounds = None# [0.2, 1.2]

t = t[(t.pl_masse > MassBounds[0]) & (t.pl_masse < MassBounds[1])]
t = t[(t.pl_rade > RadiusBounds[0]) & (t.pl_rade < RadiusBounds[1])]
# t = t[t.pl_eqt < 950]


if InsolationBounds is not None:
	t = t[(t.pl_insol > InsolationBounds[0]) & (t.pl_insol < InsolationBounds[1])]
	
if StellarMassBounds is not None:
	t = t[(t.st_mass > StellarMassBounds[0]) & (t.st_mass < StellarMassBounds[1])]
	

RemovePlanets = ['Kepler-54 b', 'Kepler-54 c']
t = t[~np.isin(t.pl_name, RemovePlanets)]

#t = t.iloc[0:20]
# t = pd.read_csv(os.path.join(DataDirectory, 'Teff_{}_ExcUpperLimits_20210316_Metallicity.csv'.format(UTeff)))
# t = pd.read_csv(os.path.join(DataDirectory, 'Teff_{}_IncUpperLimits_20210316_Metallicity.csv'.format(UTeff)))
# t = pd.read_csv(os.path.join(DataDirectory, 'Teff_{}_IncUpperLimits_20210316.csv'.format(UTeff)))



# t = pd.read_csv(os.path.join(DataDirectory, 'Teff_4400_IncUpperLimits_20210127.csv'))
# t = pd.read_csv(os.path.join(DataDirectory, 'Teff_4400_IncUpperLimits_20210127_Metallicity.csv'))
# t = pd.read_csv(os.path.join(DataDirectory, 'Teff_6000_ExcUpperLimits_20210216.csv'))

# t = t[t['st_mass'] < 10]
# t= t[t['pl_hostname'] != 'TRAPPIST-1']
# t = t[t['pl_masse'] < 50]
# t = t[t['pl_rade'] < 4]
# t = t[t['pl_rade'] > 0.8]
# t = pd.read_csv(os.path.join(pwd, 'Kepler_MR_inputs.csv'))
print(len(t))
# t = t[np.isfinite(t['pl_insol'])]
# t = t[t['pl_insol'] > 1e-10]

# t = Table.read(os.path.join(pwd,'Kepler_MR_inputs.csv'))
# t = Table.read(os.path.join(pwd,'FGK_20190406.csv'))


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


"""
#############################
### Fake Mass, Radius, Insolation
#############################

Radius = 10**np.linspace(-0.3, 2.5)
# Symmetrical errorbars
Radius_sigma1 = Radius*0.1
Radius_sigma2 = Radius*0.1
RadiusBounds = [0.1, 300]

Mass = Radius*10
# Symmetrical errorbars
Mass_sigma1 = Mass*0.1
Mass_sigma2 = Mass*0.1
MassBounds = [1, 3000]

Insolation = np.ones(len(Radius))
InsolationSigma = np.ones(len(Radius))*0.1
InsolationBounds = [0.9, 1.1]

StellarMass = np.ones(len(Radius))*0.5
StellarMassSigma = StellarMass*0.2
StellarMassBounds = [0.25, 0.75]


#############################
#############################
#############################
"""


"""
# Run with 100 bootstraps. Selecting degrees to be 17. Alternatively can set select_deg = 'cv' to
# find the optimum number of degrees.

FakePeriod = np.ones(len(Period))
# FakePeriod = Radius
NPoints = len(Radius)
# FakePeriod = np.concatenate([np.random.lognormal(0, 1, NPoints//2), \
	# np.random.lognormal(5, 1, NPoints//2+1)])

# FakePeriodSigma = FakePeriod*0.01
# Period_sigma = FakePeriodSigma


# Simulation
# Radius = 10**np.linspace(0, 1)
Radius_sigma1 = 0.1*Radius# np.repeat(np.nan, len(Radius))
Radius_sigma2 = np.copy(Radius_sigma1)
Mass = 10**(2*np.log10(Radius)*np.log10(Radius) - 0.5*np.log10(Radius))
# Mass =  np.ones(len(Radius))/2
Mass_sigma1 = np.repeat(np.nan, len(Radius))
Mass_sigma2 = np.repeat(np.nan, len(Radius))
Period = np.ones(len(Radius))/2
StellarMass_Sigma1 = np.repeat(np.nan, len(Radius))
# Period = Radius# np.linspace(0, 1, len(Radius))
PeriodSigma = 0.1*Period #np.repeat(np.nan, len(Radius))

"""

Max, Min = 1, 0
Max, Min = np.nan, np.nan
# Max += 0.2
# Min -= 0.2


RadiusDict = {'Data': Radius, 'LSigma': RadiusLSigma,  "USigma":RadiusUSigma, 'Max':np.log10(RadiusBounds[1]), 'Min':np.log10(RadiusBounds[0]), 'Label':'Radius ($R_{\oplus}$)', 'Char':'r'}
MassDict = {'Data': Mass, 'LSigma': MassLSigma, "USigma":MassUSigma, 'Max':np.log10(MassBounds[1]), 'Min':np.log10(MassBounds[0]), 'Label':'Mass ($M_{\oplus}$)', 'Char':'m'}

#FakePeriodDict = {'Data': FakePeriod, 'LSigma': PeriodSigma, "USigma":PeriodSigma, 'Max':np.nan, 'Min':np.nan, 'Label':'Period (d)', 'Char':'p'}
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

InputDictionaries = [RadiusDict, MassDict]
# InputDictionaries = [RadiusDict, MassDict, InsolationDict, StellarMassDict]
DataDict = InputData(InputDictionaries)

ndim = len(InputDictionaries)




for d in [20]:#, 40, 80, 100, 500, 1000]:
	# print(d)

	# RunName = 'Kepler_127_M_R_bounded'
	RunName = 'Mdwarf_3D_20220409_M_R_S_bounded'
	RunName = 'Fake_4D_MRSStM'
	RunName = 'GiantPlanet_d60_Bootstrap100_4D_MRSStM'
	RunName = 'Test_2d_MR_yesSparse'

	# save_path = os.path.join(pwd, 'TestRuns', 'Mdwarf_4D_20220325_M_R_S_StM')
	save_path = os.path.join(pwd, 'TestRuns',  RunName)
	 
	# deg_per_dim = [25, 25, 25, 30]


	# outputs, _ = fit_relation(DataDict, select_deg=34, save_path=save_path, NumBootstrap=0, degree_max=15)

	select_deg = [60, 60]

	if __name__ == '__main__':

		outputs= fit_relation(DataDict, select_deg=select_deg, save_path=save_path, degree_max=120, cores=25,SymmetricDegreePerDimension=True, NumMonteCarlo=0, NumBootstrap=0)
		#cProfile.run("outputs= fit_relation(DataDict, select_deg=select_deg, save_path=save_path, degree_max=120, cores=40, SymmetricDegreePerDimension=True, NumMonteCarlo=0, NumBootstrap=100)", os.path.join(save_path, 'Profile.prof'))

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

