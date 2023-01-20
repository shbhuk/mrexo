import os, sys
from astropy.table import Table
import numpy as np
from multiprocessing import cpu_count
import numpy as np


from mrexo import fit_xy_relation
from mrexo import predict_from_measurement
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


DataDirectory = os.path.join(HomeDir, 'Mdwarf-Exploration', 'Data', 'MdwarfPlanets')


UTeff = 6500

t = pd.read_csv(os.path.join(DataDirectory, 'Teff_{}_ExcUpperLimits_20210823.csv'.format(UTeff)))
t = pd.read_csv(os.path.join(DataDirectory, 'Teff_4000_ExcUpperLimits_20210910GiantPlanets_HPF_Metallicity.csv')) # HPF Giant planets
t = pd.read_csv(os.path.join(DataDirectory, 'Teff_4000_ExcUpperLimits_20210910GiantPlanets_Metallicity.csv')) # w/o HPF Giant planets
# t = pd.read_csv(os.path.join(DataDirectory, 'Teff_4000_IncUpperLimits_20210910GiantPlanets_RV_Metallicity.csv')) # RV Giant planets
print(len(t))

# No M dwarfs for this simulation
# t = t[t.st_teff < 4000]
print(len(t))

# t = t[t.pl_rade > 4]
# print(len(t))

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
Mass_sigma2 = np.array(abs(t['pl_masseerr1']))

# In Earth units
Msini = np.array(t['pl_msinie'])
# Symmetrical errorbars
Msini_sigma1 = np.array(abs(t['pl_msinieerr1']))
Msini_sigma2 = np.array(abs(t['pl_msinieerr2']))

Radius = np.array(t['pl_rade'])
# Symmetrical errorbars
Radius_sigma1 = np.array(abs(t['pl_radeerr1']))
Radius_sigma2 = np.array(abs(t['pl_radeerr1']))

# """
Period = np.array(t['pl_orbper'])
PeriodSigma = np.repeat(np.nan, len(Period))

StellarMass = np.array(t['st_mass'])
StellarMassSigma = np.array(t['st_masserr1'])

Metallicity = np.array(t['st_met'])
MetallicitySigma = np.array(t['st_meterr1'])

Insolation = np.array(t['pl_insol'])
InsolationSigma = np.array(t['pl_insolerr1'])

Density = np.array(t['pl_dens'])
DensitySigma = np.array(t['pl_denserr1'])
# """


# Directory to store results in
# result_dir = os.path.join(pwd, 'TestRuns','Mdwarfs_20200520_cv50')
# result_dir = os.path.join(pwd,'TestRuns', 'Kepler127_aic')
# result_dir = os.path.join(pwd, 'FGK_319_cv100')

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


RadiusDict = {'Data': Radius, 'SigmaLower': Radius_sigma1,  "SigmaUpper":Radius_sigma2, 'Max':Max, 'Min':Min, 'Label':'Radius ($R_{\oplus}$)', 'Char':'r'}
MassDict = {'Data': Mass, 'SigmaLower': Mass_sigma1, "SigmaUpper":Mass_sigma2, 'Max':Max, 'Min':np.nan, 'Label':'Mass ($M_{\oplus}$)', 'Char':'m'}
MsiniDict = {'Data': Msini, 'SigmaLower': Msini_sigma1, "SigmaUpper":Msini_sigma2, 'Max':Max, 'Min':np.nan, 'Label':'Msini ($M_{\oplus}$)', 'Char':'msini'}

#FakePeriodDict = {'Data': FakePeriod, 'SigmaLower': PeriodSigma, "SigmaUpper":PeriodSigma, 'Max':np.nan, 'Min':np.nan, 'Label':'Period (d)', 'Char':'p'}
PeriodDict = {'Data': Period, 'SigmaLower': PeriodSigma, "SigmaUpper":PeriodSigma, 'Max':Max, 'Min':Min, 'Label':'Period (d)', 'Char':'p'}
StellarMassDict = {'Data': StellarMass, 'SigmaLower': StellarMassSigma, "SigmaUpper":StellarMassSigma, 'Max':np.nan, 'Min':np.nan, 'Label':'Stellar Mass (M$_{\odot}$)', 'Char':'stm'}
MetallicityDict = {'Data': 10**Metallicity, 'SigmaLower': np.repeat(np.nan, len(Metallicity)), "SigmaUpper":np.repeat(np.nan, len(Metallicity)), 'Max':np.nan, 'Min':np.nan, 'Label':'Metallicity [Fe/H]', 'Char':'feh'}
# MetallicityDict = {'Data': 10**Metallicity, 'SigmaLower': np.repeat(np.nan, len(Metallicity)), "SigmaUpper":np.repeat(np.nan, len(Metallicity)), 'Max':1, 'Min':-0.45, 'Label':'Metallicity [Fe/H]', 'Char':'feh'}
# MetallicityDict = {'Data': 10**(-Metallicity), 'SigmaLower': np.repeat(np.nan, len(Metallicity)), "SigmaUpper":np.repeat(np.nan, len(Metallicity)), 'Max':np.nan, 'Min':np.nan, 'Label':'Metallicity [Fe/H]', 'Char':'feh'}

DensityDict = {'Data': Density, 'SigmaLower':DensitySigma, 'SigmaUpper':DensitySigma, 'Max':np.nan, 'Min':np.nan, 'Label':'Pl Density (g/cm3)', 'Char':'dens'}
InsolationDict = {'Data': Insolation, 'SigmaLower': InsolationSigma, "SigmaUpper":InsolationSigma, 'Max':np.nan, 'Min':np.nan, 'Label':'Pl Insol ($S_{\oplus}$)', 'Char':'insol'}

from mrexo.mle_utils_nd import InputData, MLE_fit
from mrexo.fit_nd import fit_relation
import matplotlib.pyplot as plt
# InputDictionaries = [RadiusDict, MassDict]
InputDictionaries = [RadiusDict, MassDict, InsolationDict, StellarMassDict]

# InputDictionaries = [RadiusDict, StellarMassDict, PeriodDict, MetallicityDict]
# InputDictionaries = [RadiusDict, StellarMassDict, PeriodDict]
# InputDictionaries = [MassDict, StellarMassDict, MetallicityDict]
InputDictionaries = [MassDict, StellarMassDict, MetallicityDict]

# InputDictionaries = [RadiusDict, MassDict, InsolationDict, StellarMassDict]

# InputDictionaries = [RadiusDict,  MassDict, FakePeriodDict]


# InputDictionaries = [RadiusDict, PeriodDict, StellarMassDict]
# InputDictionaries = [RadiusDict, PeriodDict, MetallicityDict]

# InputDictionaries = [RadiusDict, InsolationDict, StellarMassDict]

DataDict = InputData(InputDictionaries)
#DataDict = np.load(r"C:\Users\shbhu\Documents\GitHub\mrexo\sample_scripts\TestRuns\SimConstantDeg40\output\other_data_products\DataDict.npy", allow_pickle=True).item()

RunName = 'MdwarfGasGiant_MStM_FeH_deg30'
save_path = os.path.join(pwd, 'TestRuns', RunName)
 
ndim = len(InputDictionaries)
deg_per_dim = [25, 25, 25, 30]
deg_per_dim = [30] * ndim
# deg_per_dim = [35, 30, 32]
"""
outputs = MLE_fit(DataDict, 
	deg_per_dim=deg_per_dim,
	save_path=save_path, OutputWeightsOnly=False, CalculateJointDist=True)
"""

# outputs, _ = fit_relation(DataDict, select_deg='aic', save_path=save_path, num_boot=0, degree_max=15)

if __name__ == '__main__':
	outputs, _ = fit_relation(DataDict, select_deg=deg_per_dim, save_path=save_path, num_boot=0, degree_max=100, cores=2)

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
		plt.imshow(np.reshape(weights , deg_per_dim).T, origin = 'left', aspect='auto')
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
		plt.ylabel("Log10 "+DataDict['ndim_label'][1]);
		plt.xlabel("Log10 "+DataDict['ndim_label'][0]);
		# plt.xlabel("Planetary Mass ($M_{\oplus}$)")
		# plt.ylabel("Planetary Radius ($R_{\oplus}$)")
		plt.xlim(DataDict['ndim_bounds'][0][0], DataDict['ndim_bounds'][0][1])
		plt.ylim(DataDict['ndim_bounds'][1][0], DataDict['ndim_bounds'][1][1])
		plt.tight_layout()
		XTicks = np.linspace(x.min(), x.max(), 5)
		# XTicks = np.log10(np.array([0.3, 1, 3, 10, 30, 100, 300]))
		YTicks = np.linspace(y.min(), y.max(), 5)
		# YTicks = np.log10(np.array([1, 3, 5, 10]))


		XLabels = np.round(10**XTicks, 1)

		YLabels = np.round(10**YTicks, 2)

		plt.xticks(XTicks, XLabels)
		plt.yticks(YTicks, YLabels)
		cbar = fig.colorbar(im, ticks=[np.min(JointDist), np.max(JointDist)], fraction=0.037, pad=0.04)
		cbar.ax.set_yticklabels(['Min', 'Max'])
		plt.tight_layout()
		plt.savefig(os.path.join(save_path, 'output', 'JointDist.png'))
		plt.close("all")
		# plt.show(block=False)
		#'''

