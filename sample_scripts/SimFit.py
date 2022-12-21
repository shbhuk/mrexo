import os, sys
from astropy.table import Table
import numpy as np
from multiprocessing import cpu_count
import numpy as np
import pandas as pd

import cProfile
import pstats

from mrexo.mle_utils_nd import InputData, MLE_fit
from mrexo.fit_nd import fit_relation
import matplotlib.pyplot as plt


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

t = pd.read_csv(os.path.join(DataDirectory, 'Teff_7000_ExcUpperLimits_20220401_Thesis.csv'))

# Try = [[], []]

for n in [50]:
	for LSigma in [0.5]:
		for USigma in [0.5]:
		
			#############################
			### Fake Mass, Radius, Insolation
			#############################

			Radius_n = 10**np.linspace(-0.3, 1.2, n)
			# Symmetrical errorbars
			Radius_sigma1 = Radius_n*LSigma
			Radius_sigma2 = Radius_n*USigma
			Radius = np.random.normal(loc=Radius_n, scale=np.average([Radius_sigma1, Radius_sigma2], axis=0))
			
			RadiusBounds = [np.nan, np.nan]#[0.1, 20]

			Mass_n = Radius_n*10
			# Symmetrical errorbars
			Mass_sigma1 = Mass_n*LSigma
			Mass_sigma2 = Mass_n*USigma
			Mass = np.random.normal(loc=Mass_n, scale=np.average([Mass_sigma1, Mass_sigma2], axis=0))

			MassBounds = [np.nan, np.nan]# [1, 250]

			Insolation = np.ones(len(Radius))
			InsolationSigma = np.ones(len(Radius))*0.1
			InsolationBounds = [0.9, 1.1]

			StellarMass = np.ones(len(Radius))*0.5
			StellarMassSigma = StellarMass*0.2
			StellarMassBounds = [0.25, 0.75]


			#############################


			Max, Min = 1, 0
			Max, Min = np.nan, np.nan

			RadiusDict = {'Data': Radius, 'LSigma': Radius_sigma1,  "USigma":Radius_sigma2, 'Max':np.log10(RadiusBounds[1]), 'Min':np.log10(RadiusBounds[0]), 'Label':'Radius ($R_{\oplus}$)', 'Char':'r'}
			MassDict = {'Data': Mass, 'LSigma': Mass_sigma1, "USigma":Mass_sigma2, 'Max':np.log10(MassBounds[1]), 'Min':np.log10(MassBounds[0]), 'Label':'Mass ($M_{\oplus}$)', 'Char':'m'}

			# StellarMassDict = {'Data': StellarMass, 'LSigma': StellarMassSigma, "USigma":StellarMassSigma, 'Max':np.log10(StellarMassBounds[1]), 'Min':np.log10(StellarMassBounds[0]), 'Label':'Stellar Mass (M$_{\odot}$)', 'Char':'stm'}
			# InsolationDict = {'Data': Insolation, 'LSigma': InsolationSigma, "USigma":InsolationSigma, 'Max':np.log10(InsolationBounds[1]), 'Min':np.log10(InsolationBounds[0]), 'Label':'Pl Insol ($S_{\oplus}$)', 'Char':'insol'}

			RunName = 'Sim_2D_N{}_USigma{}_LSigma{}_aic_asymm'.format(n, USigma, LSigma)
			print(RunName)
	
			InputDictionaries = [RadiusDict, MassDict]#, StellarMassDict]
			DataDict = InputData(InputDictionaries)

			save_path = os.path.join(pwd, 'TestRuns',  RunName)
			 
			ndim = len(InputDictionaries)

			if __name__ == '__main__':

				cProfile.run("outputs, _ = fit_relation(DataDict, select_deg='aic', save_path=save_path, num_boot=0, degree_max=100, cores=4, SymmetricDegreePerDimension=False)", os.path.join(save_path, 'Profile.prof'))

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
					plt.figure()
					plt.imshow(np.reshape(weights , deg_per_dim).T, origin = 'lower', aspect='auto')

					plt.title(deg_per_dim)
					# plt.imshow(weights.reshape(deg_per_dim))
					plt.colorbar()
					plt.savefig(os.path.join(save_path, 'output', 'weights.png'))
					plt.close('all')


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
					
					XTicks = [1, 3, 8, 10, 15]
					YTicks = [5, 10, 30, 80, 100, 200]
					
					XLabels = XTicks
					YLabels = YTicks

					plt.xticks(np.log10(XTicks), XLabels)
					plt.yticks(np.log10(YTicks), YLabels)
					cbar = fig.colorbar(im, ticks=[np.min(JointDist), np.max(JointDist)], fraction=0.037, pad=0.04)
					cbar.ax.set_yticklabels(['Min', 'Max'])
					plt.tight_layout()
					plt.savefig(os.path.join(save_path, 'output', 'JointDist.png'))
					plt.close("all")


