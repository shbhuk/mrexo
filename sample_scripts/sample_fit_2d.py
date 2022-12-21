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


try :
	pwd = os.path.dirname(__file__)
except NameError:
	pwd = os.path.join(HomeDir, 'mrexo', 'sample_scripts')
	print('Could not find pwd')


DataDirectory = os.path.join(HomeDir, 'Mdwarf-Exploration', 'Data', 'MdwarfPlanets')


UTeff = 7000

# t = pd.read_csv(os.path.join(DataDirectory, 'Teff_{}_ExcUpperLimits_20210823.csv'.format(UTeff)))
# t = pd.read_csv(os.path.join(DataDirectory, 'Kepler_Teff_6500_ExcUpperLimits_20210908.csv'))
t = pd.read_csv(os.path.join(DataDirectory, 'Teff_7000_ExcUpperLimits_20220401_Thesis.csv'))
# t = pd.read_csv(os.path.join(pwd, 'Cool_stars_20181214_exc_upperlim.csv'))
# t = pd.read_csv(os.path.join(pwd, 'Kepler_MR_inputs.csv'))

RadiusBounds = [4, 15]
MassBounds = [4, 3000]
InsolationBounds = [0.01, 5000]
StellarMassBounds = [0.2, 1.2]

t = t[(t.pl_masse > MassBounds[0]) & (t.pl_masse < MassBounds[1])]
t = t[(t.pl_rade > RadiusBounds[0]) & (t.pl_rade < RadiusBounds[1])]

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
Mass_sigma1 = np.array(abs(t['pl_masseerr1']))
Mass_sigma2 = np.array(abs(t['pl_masseerr2']))

Radius = np.array(t['pl_rade'])
# Symmetrical errorbars
Radius_sigma1 = np.array(abs(t['pl_radeerr1']))
Radius_sigma2 = np.array(abs(t['pl_radeerr2']))

"""
Period = np.array(t['pl_orbper'])
PeriodSigma = np.repeat(np.nan, len(Period))

"""


StellarMass = np.array(t['st_mass'])
StellarMassSigma = np.array(t['st_masserr1'])

# Metallicity = np.array(t['st_met'])
# MetallicitySigma = np.array(t['st_meterr1'])

Insolation = np.array(t['pl_insol'])
InsolationSigma = np.array(t['pl_insolerr1'])

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


RadiusDict = {'Data': Radius, 'LSigma': Radius_sigma1,  "USigma":Radius_sigma2, 'Max':np.log10(RadiusBounds[1]), 'Min':np.log10(RadiusBounds[0]), 'Label':'Radius ($R_{\oplus}$)', 'Char':'r'}
MassDict = {'Data': Mass, 'LSigma': Mass_sigma1, "USigma":Mass_sigma2, 'Max':np.log10(MassBounds[1]), 'Min':np.log10(MassBounds[0]), 'Label':'Mass ($M_{\oplus}$)', 'Char':'m'}

#FakePeriodDict = {'Data': FakePeriod, 'LSigma': PeriodSigma, "USigma":PeriodSigma, 'Max':np.nan, 'Min':np.nan, 'Label':'Period (d)', 'Char':'p'}
# PeriodDict = {'Data': Period, 'LSigma': PeriodSigma, "USigma":PeriodSigma, 'Max':Max, 'Min':Min, 'Label':'Period (d)', 'Char':'p'}
StellarMassDict = {'Data': StellarMass, 'LSigma': StellarMassSigma, "USigma":StellarMassSigma, 'Max':np.log10(StellarMassBounds[1]), 'Min':np.log10(StellarMassBounds[0]), 'Label':'Stellar Mass (M$_{\odot}$)', 'Char':'stm'}
#MetallicityDict = {'Data': 10**Metallicity, 'LSigma': np.repeat(np.nan, len(Metallicity)), "USigma":np.repeat(np.nan, len(Metallicity)), 'Max':np.nan, 'Min':np.nan, 'Label':'Metallicity [Fe/H]', 'Char':'feh'}
# MetallicityDict = {'Data': 10**Metallicity, 'LSigma': np.repeat(np.nan, len(Metallicity)), "USigma":np.repeat(np.nan, len(Metallicity)), 'Max':1, 'Min':-0.45, 'Label':'Metallicity [Fe/H]', 'Char':'feh'}
# MetallicityDict = {'Data': 10**(-Metallicity), 'LSigma': np.repeat(np.nan, len(Metallicity)), "USigma":np.repeat(np.nan, len(Metallicity)), 'Max':np.nan, 'Min':np.nan, 'Label':'Metallicity [Fe/H]', 'Char':'feh'}

InsolationDict = {'Data': Insolation, 'LSigma': InsolationSigma, "USigma":InsolationSigma, 'Max':np.log10(InsolationBounds[1]), 'Min':np.log10(InsolationBounds[0]), 'Label':'Pl Insol ($S_{\oplus}$)', 'Char':'insol'}

degree_candidates = np.loadtxt( r"/storage/home/szk381/work/mrexo/sample_scripts/TestRuns/Trial_FGKM_2D_MR_aic_asymm_degmin10_20c/output/other_data_products/degree_candidates.txt")


from mrexo.mle_utils_nd import InputData, MLE_fit
from mrexo.fit_nd import fit_relation
import matplotlib.pyplot as plt

for xdeg in degree_candidates[0].astype(int):
	for ydeg in degree_candidates[0].astype(int):

		# RunName = 'Kepler_127_M_R_bounded'
		RunName = 'Mdwarf_3D_20220409_M_R_S_bounded'
		RunName = 'Fake_4D_MRSStM'
		RunName = 'Trial_FGKM_2D_MR_{}_{}'.format(xdeg, ydeg)
		select_deg = [xdeg, ydeg]
		print(RunName)

		# InputDictionaries = [RadiusDict, MassDict, InsolationDict]
		# InputDictionaries = [RadiusDict, MassDict, InsolationDict, StellarMassDict]

		# InputDictionaries = [RadiusDict, StellarMassDict, PeriodDict, MetallicityDict]
		# InputDictionaries = [RadiusDict, StellarMassDict, PeriodDict]
		# InputDictionaries = [RadiusDict, InsolationDict]
		InputDictionaries = [RadiusDict,  MassDict]


		# InputDictionaries = [RadiusDict, PeriodDict, StellarMassDict]
		# InputDictionaries = [RadiusDict, PeriodDict, MetallicityDict]

		# InputDictionaries = [RadiusDict, InsolationDict, StellarMassDict]

		DataDict = InputData(InputDictionaries)

		# save_path = os.path.join(pwd, 'TestRuns', 'Mdwarf_4D_20220325_M_R_S_StM')
		save_path = os.path.join(pwd, 'TestRuns', 'FGKM_2D_MR', RunName)
		 
		ndim = len(InputDictionaries)
		# deg_per_dim = [25, 25, 25, 30]

		"""
		outputs = MLE_fit(DataDict, 
			deg_per_dim=deg_per_dim,
			save_path=save_path, OutputWeightsOnly=False, CalculateJointDist=True)

		from mrexo.utils import _logging
		from multiprocessing import Pool

		verbose=2
		select_deg = 'cv'
		num_boot=0
		degree_max=40
		cores=1
		k_fold=10
		NumCandidates=10
		"""

		# outputs, _ = fit_relation(DataDict, select_deg=34, save_path=save_path, num_boot=0, degree_max=15)

		if __name__ == '__main__':

			cProfile.run("outputs, _ = fit_relation(DataDict, select_deg=select_deg, save_path=save_path, num_boot=0,degree_max=60, cores=4, SymmetricDegreePerDimension=False)", os.path.join(save_path, 'Profile.prof'))

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

			JointDist = outputs['JointDist']
			weights = outputs['Weights']
			unpadded_weight = outputs['UnpaddedWeights']
			deg_per_dim = outputs['deg_per_dim']

			for n in range(ndim):
				x = DataDict['ndim_data'][n]
				plt.hist(x, bins=np.logspace(np.log10(x.min()), np.log10(x.max()), 20+1))
				plt.xlabel(DataDict['ndim_label'][n])
				plt.gca().set_xscale("log")
				plt.title(RunName)
				plt.tight_layout()
				plt.savefig(os.path.join(save_path, 'output', 'Histogram_'+DataDict['ndim_char'][n]+'.png'))
				plt.close("all")

			if ndim==2:
					
				# C_pdf_new = np.loadtxt(os.path.join(save_path, 'C_pdf.txt'))
				# plt.figure()
				# plt.imshow(C_pdf_new, aspect='auto')
				# plt.title(deg_per_dim)
				# plt.show(block=False)

				plt.figure()
				plt.imshow(np.reshape(weights , deg_per_dim).T, origin = 'lower', aspect='auto')
				# plt.xticks(np.arange(0,size), *[np.arange(0,size)])
				# plt.yticks(np.arange(0,size), *[np.arange(0,size)])
				plt.title(deg_per_dim)
				# plt.imshow(weights.reshape(deg_per_dim))
				plt.colorbar()
				plt.savefig(os.path.join(save_path, 'output', 'weights.png'))
				plt.close('all')
				# plt.show(block=False)


				# plt.figure()
				# plt.imshow(np.reshape(unpadded_weight, np.array(deg_per_dim)-2).T, origin='lower')
				# plt.show(block=False)

				################ Plot Joint Distribution ################ 
				x = DataDict['DataSequence'][0]
				y = DataDict['DataSequence'][1]

				fig = plt.figure(figsize=(8.5,6.5))
				im = plt.imshow(JointDist.T, 
					extent=(DataDict['ndim_bounds'][0][0], DataDict['ndim_bounds'][0][1], DataDict['ndim_bounds'][1][0], DataDict['ndim_bounds'][1][1]), 
					aspect='auto', origin='lower'); 
				plt.errorbar(x=np.log10(DataDict['ndim_data'][0]), y=np.log10(DataDict['ndim_data'][1]), xerr=0.434*DataDict['ndim_sigma'][0]/DataDict['ndim_data'][0], yerr=0.434*DataDict['ndim_sigma'][1]/DataDict['ndim_data'][1], fmt='.', color='k', alpha=0.4)
				# plt.title("Orbital Period = {} d".format(str(np.round(title,3))))
				plt.ylabel(DataDict['ndim_label'][1]);
				plt.xlabel(DataDict['ndim_label'][0]);
				# plt.xlabel("Planetary Mass ($M_{\oplus}$)")
				# plt.ylabel("Planetary Radius ($R_{\oplus}$)")
				plt.xlim(DataDict['ndim_bounds'][0][0], DataDict['ndim_bounds'][0][1])
				plt.ylim(DataDict['ndim_bounds'][1][0], DataDict['ndim_bounds'][1][1])
				plt.tight_layout()
				
				# """
				XTicks = np.linspace(x.min(), x.max(), 5)
				XTicks = np.log10(np.array([3, 10, 30, 100, 300]))
				YTicks = np.linspace(y.min(), y.max(), 5)
				YTicks = np.log10(np.array([1, 3, 5, 10]))
				XLabels = np.round(10**XTicks, 1)
				YLabels = np.round(10**YTicks, 2)
				# """
				
				XTicks = [4, 6, 8, 10, 15]
				YTicks = [3, 10, 30, 100, 300, 1000]
				
				XLabels = XTicks
				YLabels = YTicks

				plt.xticks(np.log10(XTicks), XLabels)
				plt.yticks(np.log10(YTicks), YLabels)
				cbar = fig.colorbar(im, ticks=[np.min(JointDist), np.max(JointDist)], fraction=0.037, pad=0.04)
				cbar.ax.set_yticklabels(['Min', 'Max'])
				plt.tight_layout()
				plt.savefig(os.path.join(save_path, 'output', 'JointDist.png'))
				plt.close("all")
				# plt.show(block=False)
				#'''

