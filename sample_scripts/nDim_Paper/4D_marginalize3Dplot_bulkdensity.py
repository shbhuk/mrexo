import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib import gridspec

import numpy as np
# import imageio
import glob, os
import pandas as pd
from mrexo.mle_utils_nd import calculate_conditional_distribution
import datetime
import gc # import garbage collector interface

from pyastrotools.general_tools import SigmaAntiLog10, SigmaLog10
from pyastrotools.astro_tools import calculate_orbperiod


ConditionString = 'm|r,insol,stm'
ConditionName = '4D_'+ConditionString.replace('|', '_').replace(',', '_')

ResultDirectory = r'C:\Users\skanodia\Documents\GitHub\mrexo\sample_scripts\TestRuns\AllPlanet_RpLt4_StMlt1.5_MRSStM_d40_100MC_100BS'
PlotFolder = os.path.join(ResultDirectory, ConditionName)
PlotFolder = os.path.join(ResultDirectory, '4D_m_r_stm_insol')

if not os.path.exists(PlotFolder):
	print("4D Plot folder does not exist")
	os.mkdir(PlotFolder)

deg_per_dim = np.loadtxt(os.path.join(ResultDirectory, 'output', 'deg_per_dim.txt'))
DataDict = np.load(os.path.join(ResultDirectory, 'input', 'DataDict.npy'), allow_pickle=True).item()
FullJointDist = np.load(os.path.join(ResultDirectory, 'output', 'JointDist.npy'), allow_pickle=True).T
Fullweights = np.genfromtxt(os.path.join(ResultDirectory, 'output', 'weights.txt'))
deg_per_dim = np.loadtxt(os.path.join(ResultDirectory, 'output', 'deg_per_dim.txt')).astype(int)

Condition = ConditionString.split('|')
LHSTerms = Condition[0].split(',')
RHSTerms = Condition[1].split(',')
deg_vec_per_dim = [np.arange(1, deg+1) for deg in deg_per_dim] 

LHSDimensions = np.array([(np.arange(DataDict['ndim'])[np.isin(DataDict['ndim_char'] , l)])[0] for l in LHSTerms])
RHSDimensions = np.array([(np.arange(DataDict['ndim'])[np.isin(DataDict['ndim_char'] , r)])[0] for r in RHSTerms])

x = DataDict['DataSequence'][RHSDimensions[0]] # pl_rade
y = DataDict['DataSequence'][RHSDimensions[1]][::2] # insol
y = np.log10([10, 50, 100, 500])
z = DataDict['DataSequence'][RHSDimensions[2]] # stm
MassAxis = DataDict['DataSequence'][LHSDimensions[0]] # Pl Mass

xdata = DataDict['ndim_data'][RHSDimensions[0]]
ydata = DataDict['ndim_data'][RHSDimensions[1]]
zdata = DataDict['ndim_data'][RHSDimensions[2]]

rho_earth = 5.514 #g/cm3
rho_neptune = 1.638 #g/cm3

Bootstrap = True
MonteCarlo = True
ConsiderRadii = np.arange(1.0, 3.01, 0.5)
# ConsiderRadii = np.arange(1.0, 2.1, 0.1)


cmap = matplotlib.cm.Spectral
cmap = matplotlib.cm.YlGnBu
# Establish colour range based on variable
norm = matplotlib.colors.Normalize(vmin=ConsiderRadii.min()/2, vmax=ConsiderRadii.max())

# [0.1, 0.3, 0.5, 0.7, 1.0, 1.2]

"""
# for st_m in [0.5]
# for st_m in [0.3, 1.2]:

for st_m in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.1]:
	print("Stellar Mass = ", st_m)

	if Bootstrap:
		BootstrapDirectory = os.path.join(ResultDirectory, 'output', 'other_data_products', 'Bootstrap')
		NumBootstrap = len(glob.glob(os.path.join(BootstrapDirectory, 'JointDist_BSSim*.npy')))
		BootstrapMean = np.zeros((len(ConsiderRadii), len(y), NumBootstrap))

	if MonteCarlo:
		MonteCarloDirectory = os.path.join(ResultDirectory, 'output', 'other_data_products', 'MonteCarlo')
		NumMonteCarlo = len(glob.glob(os.path.join(MonteCarloDirectory, 'JointDist_MCSim*.npy')))
		MonteCarloMean = np.zeros((len(ConsiderRadii), len(y), NumMonteCarlo))

	FullSampleMean = np.zeros((len(ConsiderRadii), len(y)))

	for i in range(len(ConsiderRadii)):
		pl_rade = ConsiderRadii[i]
		print("		Planetary Radius = ", pl_rade, datetime.datetime.now())

		MeanPDF = np.zeros(len(y))
		
		for j in range(len(y)):
			pl_insol = y[j]
			print("			Planetary Insolation = ", pl_insol)
				
			MeasurementDict = {
				RHSTerms[0]:[[np.log10(pl_rade)], [[np.nan, np.nan]]], 
				RHSTerms[1]:[[pl_insol], [[np.nan, np.nan]]],
				RHSTerms[2]:[[np.log10(st_m)], [[np.nan, np.nan]]]}
				
			# Conditonal Distribution of Mass 
			ConditionalDist, MeanPDF[j], _ = calculate_conditional_distribution(
				ConditionString, DataDict, Fullweights, deg_per_dim,
				FullJointDist.T, MeasurementDict)
				


			if Bootstrap:
				print("				Running Bootstrap @ ", datetime.datetime.now())
				for b in range(NumBootstrap):
					BSJointDist = np.load(os.path.join(BootstrapDirectory, 'JointDist_BSSim{}.npy'.format(int(b))), allow_pickle=True).T
					BSweights = np.asarray(pd.read_csv(os.path.join(BootstrapDirectory, 'weights_BSSim{}.txt'.format(int(b))))).reshape(len(Fullweights))
					_, BSMass, _= calculate_conditional_distribution(
						ConditionString, DataDict, BSweights, deg_per_dim,
						BSJointDist.T, MeasurementDict)

					del BSJointDist, BSweights
					gc.collect()

					BootstrapMean[i, j, b] = (10**BSMass / (pl_rade**3)) * rho_earth

			if MonteCarlo:
				print("				Running MonteCarlo @ ", datetime.datetime.now())
				for b in range(NumMonteCarlo):
					MCJointDist = np.load(os.path.join(MonteCarloDirectory, 'JointDist_MCSim{}.npy'.format(int(b))), allow_pickle=True).T
					MCweights = np.asarray(pd.read_csv(os.path.join(MonteCarloDirectory, 'weights_MCSim{}.txt'.format(int(b))))).reshape(len(Fullweights))
					_, MCMass, _= calculate_conditional_distribution(
						ConditionString, DataDict, MCweights, deg_per_dim,
						MCJointDist.T, MeasurementDict)
					
					del MCJointDist, MCweights
					gc.collect()

					MonteCarloMean[i, j, b] = (10**MCMass / (pl_rade**3)) * rho_earth

		Density_Earth = (10**MeanPDF) / pl_rade**3
		Density_gcm3 = Density_Earth  * rho_earth
		FullSampleMean[i] = Density_gcm3


		plt.plot(10**y, FullSampleMean[i], c=cmap(norm(pl_rade)), label= "{:.2f} ".format(pl_rade) +  "$R_{\oplus}$", lw=5)
		if Bootstrap:
			plt.fill_between(10**y, np.percentile(BootstrapMean[i], 16, axis=1), np.percentile(BootstrapMean[i], 84, axis=1),  color=cmap(norm(pl_rade)), alpha=0.1)

		if MonteCarlo:
			plt.fill_between(10**y, np.percentile(MonteCarloMean[i], 16, axis=1), np.percentile(MonteCarloMean[i], 84, axis=1),  color=cmap(norm(pl_rade)), alpha=0.3)


	plt.axhline(rho_earth, c='k', ls='--', label=r'$\rho$ Earth')
	plt.axhline(rho_neptune, c='k', ls='-.', alpha=0.5, label=r'$\rho$ Neptune')

	plt.legend(prop={'size': 18}, loc=1)
	plt.xlabel('Insolation ($S_{\oplus}$)', size=25)
	plt.ylabel('Pl Density (g/cm3)', size=25)
	plt.xscale("log")
	plt.xlim(220, 0.5)
	plt.ylim(1, 15)
	plt.xticks(fontsize=24)
	plt.yticks(fontsize=24)
	plt.title("Stellar Mass = {:.1f}".format(st_m) + "$M_{\odot}$", size=28)#\n(pl_mass|pl_rade, pl_insol, st_m)")
	plt.tight_layout()

	PlotName = 'BulkDensity_StM{:.1f}'.format(st_m)

	if MonteCarlo: PlotName+= '_MC'
	if Bootstrap: PlotName += '_BS'

	plt.savefig(os.path.join(PlotFolder, PlotName+'.png'), dpi=360)
	plt.show(block=False)
	plt.close("all")

	if MonteCarlo: np.save(os.path.join(PlotFolder, 'BulkDensity_StM{:.1f}_MC.npy'.format(st_m)), MonteCarloMean)
	if Bootstrap: np.save(os.path.join(PlotFolder,'BulkDensity_StM{:.1f}_BS.npy'.format(st_m)), BootstrapMean)
	np.save(os.path.join(PlotFolder,'BulkDensity_StM{:.1f}_Full.npy'.format(st_m)), FullSampleMean)

"""

# plt.savefig(os.path.join(r"C:\Users\shbhu\Documents\GitHub\PostDoc\Plots", "BulkDensity_StM{:.1f}_noTRAPPIST.png".format(st_m)))
# plt.savefig(os.path.join(r"C:\Users\shbhu\Documents\GitHub\PostDoc\Plots", "BulkDensity_StM{:.1f}.png".format(st_m)))
# plt.savefig(os.path.join(r"C:\Users\shbhu\Documents\GitHub\PostDoc\Plots", "BulkDensity_StM{:.1f}_FGKM.png".format(st_m)))

"""
pl_rade = 2
pl_insol = 1 # log10(1) = 10 S_earth
# st_m = 1
MeasurementDict = {
	RHSTerms[0]:[[np.log10(pl_rade)], [np.nan]], 
	RHSTerms[1]:[[pl_insol], [np.nan]],
	RHSTerms[2]:[[np.log10(st_m)], [np.nan]]}

ConditionalDist, m, _ = calculate_conditional_distribution(
	ConditionString, DataDict, weights, deg_per_dim,
	JointDist.T, MeasurementDict)


plt.figure()
plt.plot(10**MassAxis, ConditionalDist[0])
plt.axvline(10**m, color='k', linestyle='dashed', label="{} M_e".format(np.round(10**m[0], 2)))
plt.legend()
plt.title("Predicted Mass at {:.0f} Earth radii and {:.1f} S_earth\n # = {}".format(pl_rade, 10**pl_insol, len(xdata)))
plt.tight_layout()
plt.show(block=False)
"""

# Density_Earth = (Mass) / pl_rade**3
# DensitySigma_Earth = MassSigma/ (pl_rade**3)
# Density_gcm3 = Density_Earth  * rho_earth

# """
################ PLOTTING ROUTINE - density vs insolation ################

StellarMasses = [0.1, 0.3, 0.5, 0.7, 1.0]
StellarLuminosity = [2e-4, 100e-4, 350e-4, 0.2, 1]

StellarMasses = [0.3, 0.5, 0.7, 1.0]
StellarLuminosity = [100e-4, 350e-4, 0.2, 1]

MonteCarlo = False


fig, axes = plt.subplots(figsize=(15, 9), sharey=True)
plt.axis('off')

spec = gridspec.GridSpec(ncols=2, nrows=2, wspace=0.05, hspace=0.35)
						 # width_ratios=[1, 1, 1], wspace=0.05,
						 # , height_ratios=[2, 1, 0.8, 2, 1, 0.8, 2, 1])

for kk, st_m in enumerate(StellarMasses):

	# plt.close("all")
	legend_elements1 = []
	legend_elements2 = []
	FullSampleMean = np.load(os.path.join(PlotFolder,'BulkDensity_StM{:.1f}_Full.npy'.format(st_m)))
	BootstrapMean = np.load(os.path.join(PlotFolder,'BulkDensity_StM{:.1f}_BS.npy'.format(st_m)))
	MonteCarloMean = np.load(os.path.join(PlotFolder,'BulkDensity_StM{:.1f}_MC.npy'.format(st_m)))
	
	axes = fig.add_subplot(spec[kk])


	for i in range(len(ConsiderRadii)):
		pl_rade = ConsiderRadii[i]

		# for j in range(len(y)):
			# pl_insol = y[j]


		axes.plot(10**y, FullSampleMean[i], c=cmap(norm(pl_rade)), lw=5) #, label= "{:.2f} ".format(pl_rade) +  "$R_{\oplus}$")
		legend_elements1.append(mpatches.Patch(color=cmap(norm(pl_rade)), label= "{:.2f} ".format(pl_rade) +  "$R_{\oplus}$"))

		if Bootstrap:
			axes.fill_between(10**y, np.percentile(BootstrapMean[i], 16, axis=1), np.percentile(BootstrapMean[i], 84, axis=1),  color=cmap(norm(pl_rade)), alpha=0.3)

		if MonteCarlo:
			axes.fill_between(10**y, np.percentile(MonteCarloMean[i], 16, axis=1), np.percentile(MonteCarloMean[i], 84, axis=1),  color=cmap(norm(pl_rade)), alpha=0.4)

	axes.axhline(rho_earth, c='k', ls='--')
	axes.axhline(rho_neptune, c='k', ls='-.', alpha=0.5)
	legend_elements2.append(Line2D([0], [0], color='k', ls = '--',  label=r'$\rho_{Earth}$'))
	legend_elements2.append(Line2D([0], [0], color='k', ls = '-.',  label=r'$\rho_{Neptune}$'))




	# plt.legend(prop={'size': 18}, loc=1)
	axes.set_xlim(500, 0.5)
	axes.set_ylim(1, 25)
	axes.grid(False, axis="both")
	# axes.grid(None)

	# axes.tick_params(labelsize=24)
	axes.tick_params(direction='in', length=6, width=2, which='both')
	# axes.set_yticks([0, 3, 6, 9, 12, 15])

	Insolation = np.array([500, 100, 10, 1])
	SemiMajorAxis = np.sqrt(StellarLuminosity[kk]/Insolation)
	Periods = np.round(calculate_orbperiod(st_mass=st_m, pl_orbsmax=SemiMajorAxis)*365, 1)
	ax22 = axes.twiny()
	axes.set_xscale("log")
	ax22.set_xscale("log")
	axes.set_yscale("log")

	ax22.set_xlim(axes.get_xlim())
	ax22.set_xticks(Insolation)
	ax22.set_xticklabels(Periods, fontsize=20)

	if kk < 2: 	ax22.set_xlabel('Period (d)', size=22, labelpad=6)
	else: axes.set_xlabel('Insolation ($S_{\oplus}$)', size=22, labelpad=-5)
	if kk%2!=0: 
		axes.set_yticklabels(())
	else: 
		axes.set_ylabel('Bulk Density (g/cm$^3$)', size=25)

	axes.text(2, 12, "{:.1f}".format(st_m) + "$M_{\odot}$", size=23, weight="bold")#\n(pl_mass|pl_rade, pl_insol, st_m)")



	PlotName = 'NewBulkDensity_StM{:.1f}'.format(st_m)

	if MonteCarlo: PlotName+= '_MC'
	if Bootstrap: PlotName += '_BS'

# axes = fig.add_subplot(spec[kk+1])
legend1 = axes.legend(handles=legend_elements1, loc=1, prop={'size': 20},  bbox_to_anchor=(0.9, 3.0), ncol=5)
axes.add_artist(legend1)
legend2 = axes.legend(handles=legend_elements2, loc=2, prop={'size': 20},  bbox_to_anchor=(-0.35, 2.78), ncol=2)
axes.add_artist(legend2)
# axes.axis("off")
# plt.tight_layout()
plt.subplots_adjust(left=0.07, bottom=0.08, top=0.75)

plt.suptitle("f (M$_p$|R$_p$, S, M$_*$)", fontsize=28, weight="bold")
plt.show(block=False)

	plt.savefig(os.path.join(PlotFolder, '4d_StMassPanel.pdf'), dpi=360)



################ PLOTTING ROUTINE - mass vs insolation ################

StellarMasses = [0.1, 0.3, 0.5, 0.7, 1.0]
StellarLuminosity = [2e-4, 100e-4, 350e-4, 0.2, 1]

StellarMasses = [0.3, 0.5, 0.7, 1.0]
StellarLuminosity = [100e-4, 350e-4, 0.2, 1]

MonteCarlo = False


fig, axes = plt.subplots(figsize=(15, 9), sharey=True)
plt.axis('off')

spec = gridspec.GridSpec(ncols=2, nrows=2, wspace=0.05, hspace=0.35)
						 # width_ratios=[1, 1, 1], wspace=0.05,
						 # , height_ratios=[2, 1, 0.8, 2, 1, 0.8, 2, 1])

for kk, st_m in enumerate(StellarMasses):

	# plt.close("all")
	legend_elements1 = []
	legend_elements2 = []
	FullSampleMean = np.load(os.path.join(PlotFolder,'BulkDensity_StM{:.1f}_Full.npy'.format(st_m)))
	BootstrapMean = np.load(os.path.join(PlotFolder,'BulkDensity_StM{:.1f}_BS.npy'.format(st_m)))
	MonteCarloMean = np.load(os.path.join(PlotFolder,'BulkDensity_StM{:.1f}_MC.npy'.format(st_m)))
	
	axes = fig.add_subplot(spec[kk])


	for i in range(len(ConsiderRadii)):
		pl_rade = ConsiderRadii[i]

		# for j in range(len(y)):
			# pl_insol = y[j]


		axes.plot(10**y, (FullSampleMean[i]/rho_earth)*pl_rade**3, c=cmap(norm(pl_rade)), lw=5) #, label= "{:.2f} ".format(pl_rade) +  "$R_{\oplus}$")
		legend_elements1.append(mpatches.Patch(color=cmap(norm(pl_rade)), label= "{:.2f} ".format(pl_rade) +  "$R_{\oplus}$"))

		if Bootstrap:
			axes.fill_between(10**y, (np.percentile(BootstrapMean[i], 16, axis=1)/rho_earth)*pl_rade**3, (np.percentile(BootstrapMean[i], 84, axis=1)/rho_earth)*pl_rade**3,  color=cmap(norm(pl_rade)), alpha=0.3)

		if MonteCarlo:
			axes.fill_between(10**y, (np.percentile(MonteCarloMean[i], 16, axis=1)/rho_earth)*pl_rade**3, (np.percentile(MonteCarloMean[i], 84, axis=1)/rho_earth)*pl_rade**3,  color=cmap(norm(pl_rade)), alpha=0.4)

	axes.axhline(1, c='k', ls='--')
	axes.axhline(17.15, c='k', ls='-.', alpha=0.5)
	legend_elements2.append(Line2D([0], [0], color='k', ls = '--',  label=r'$M_{\oplus}$'))
	legend_elements2.append(Line2D([0], [0], color='k', ls = '-.',  label=r'$M_{Neptune}$'))




	axes.set_xlim(500, 0.5)
	# axes.set_ylim(1, 15)
	axes.grid(False, axis="both")
	# axes.grid(None)

	# axes.tick_params(labelsize=24)
	# axes.tick_params(direction='in', length=6, width=2, which='both')
	# axes.set_yticks([0, 3, 6, 9, 12, 15])

	Insolation = np.array([500, 100, 10, 1])
	SemiMajorAxis = np.sqrt(StellarLuminosity[kk]/Insolation)
	Periods = np.round(calculate_orbperiod(st_mass=st_m, pl_orbsmax=SemiMajorAxis)*365, 1)
	ax22 = axes.twiny()
	axes.set_xscale("log")
	ax22.set_xscale("log")
	axes.set_yscale("log")

	ax22.set_xlim(axes.get_xlim())
	ax22.set_xticks(Insolation)
	ax22.set_xticklabels(Periods, fontsize=18)

	if kk < 2: 	ax22.set_xlabel('Period (d)', size=22, labelpad=6)
	else: axes.set_xlabel('Insolation ($S_{\oplus}$)', size=22, labelpad=-5)
	if kk%2!=0: 
		axes.set_yticklabels(())
	else: 
		axes.set_ylabel('Planet Mass ($M_{\oplus}$)', size=25)

	axes.text(2, 19, "{:.1f}".format(st_m) + "$M_{\odot}$", size=23, weight="bold")#\n(pl_mass|pl_rade, pl_insol, st_m)")



	PlotName = 'NewBulkDensity_StM{:.1f}'.format(st_m)

	if MonteCarlo: PlotName+= '_MC'
	if Bootstrap: PlotName += '_BS'

# axes = fig.add_subplot(spec[kk+1])
legend1 = axes.legend(handles=legend_elements1, loc=1, prop={'size': 20},  bbox_to_anchor=(0.9, 3.0), ncol=5)
axes.add_artist(legend1)
legend2 = axes.legend(handles=legend_elements2, loc=2, prop={'size': 20},  bbox_to_anchor=(-0.35, 2.79), ncol=2)
axes.add_artist(legend2)
# axes.axis("off")
# plt.tight_layout()
plt.subplots_adjust(left=0.07, bottom=0.08, top=0.75)

plt.suptitle("f (M$_p$|R$_p$, S, M$_*$)", fontsize=28, weight="bold")
plt.show(block=False)

plt.savefig(os.path.join(PlotFolder, '4d_StMass_PlMassPanel.pdf'), dpi=360)


################ PLOTTING ROUTINE - density vs stellar mass ################

StellarMasses = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.1]
# StellarLuminosity = [100e-4, 350e-4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]

ChosenInsolation = [10, 50, 100, 500]
# ChosenInsolationIndex = [9, 15] 
ChosenInsolationIndex = [0, 1, 2, 3] 

MonteCarlo = False

FullArray = np.zeros((len(StellarMasses),  len(ConsiderRadii), len(ChosenInsolation)))
BootstrapArray = np.zeros((len(StellarMasses), len(ConsiderRadii), len(ChosenInsolation), 100))

for kk, st_m in enumerate(StellarMasses):

	FullSampleMean = np.load(os.path.join(PlotFolder,'BulkDensity_StM{:.1f}_Full.npy'.format(st_m)))
	BootstrapMean = np.load(os.path.join(PlotFolder,'BulkDensity_StM{:.1f}_BS.npy'.format(st_m)))
	MonteCarloMean = np.load(os.path.join(PlotFolder,'BulkDensity_StM{:.1f}_MC.npy'.format(st_m)))

	FullArray[kk] = FullSampleMean[:, ChosenInsolationIndex]
	BootstrapArray[kk] = BootstrapMean[:, ChosenInsolationIndex]



fig, axes = plt.subplots(figsize=(15, 9), sharey=True)
plt.axis('off')

spec = gridspec.GridSpec(ncols=2, nrows=2, wspace=0.05, hspace=0.35)
						 # width_ratios=[1, 1, 1], wspace=0.05,
						 # , height_ratios=[2, 1, 0.8, 2, 1, 0.8, 2, 1])


for j in range(len(ChosenInsolation)):

	legend_elements1 = []
	legend_elements2 = []

	axes = fig.add_subplot(spec[j])


	for i in range(len(ConsiderRadii)):
		pl_rade = ConsiderRadii[i]

		axes.plot(StellarMasses, FullArray[:, i, j], c=cmap(norm(pl_rade)), lw=5) #, label= "{:.2f} ".format(pl_rade) +  "$R_{\oplus}$")
		legend_elements1.append(mpatches.Patch(color=cmap(norm(pl_rade)), label= "{:.2f} ".format(pl_rade) +  "$R_{\oplus}$"))

		if Bootstrap:
			axes.fill_between(StellarMasses, np.percentile(BootstrapArray[:, i, j], 16, axis=1), np.percentile(BootstrapArray[:, i, j], 84, axis=1),  color=cmap(norm(pl_rade)), alpha=0.3)

	axes.axhline(rho_earth, c='k', ls='--')
	axes.axhline(rho_neptune, c='k', ls='-.', alpha=0.5)
	legend_elements2.append(Line2D([0], [0], color='k', ls = '--',  label=r'$\rho_{Earth}$'))
	legend_elements2.append(Line2D([0], [0], color='k', ls = '-.',  label=r'$\rho_{Neptune}$'))

	# plt.legend(prop={'size': 18}, loc=1)
	axes.set_xlim(0.18, 1.1)
	axes.set_ylim(1, 15)
	axes.grid(False, axis="both")
	# axes.grid(None)

	# axes.tick_params(labelsize=24)
	axes.tick_params(direction='in', length=6, width=2, which='both')
	axes.set_yticks([0, 3, 6, 9, 12, 15])

	if j%2!=0: 
		axes.set_yticklabels(())
	else: 
		axes.set_ylabel('Bulk Density (g/cm$^3$)', size=25)

	axes.text(0.3, 12, "{:.0f}".format(ChosenInsolation[j]) + "$S_{\oplus}$", size=23, weight="bold")#\n(pl_mass|pl_rade, pl_insol, st_m)")

	# axes.set_ylabel('Bulk Density (g/cm$^3$)', size=25)
	axes.set_xlabel("Stellar Mass ($M_{\odot}$)")

	PlotName = 'NewBulkDensity_StM{:.1f}'.format(st_m)

	if MonteCarlo: PlotName+= '_MC'
	if Bootstrap: PlotName += '_BS'

# axes = fig.add_subplot(spec[kk+1])
legend1 = axes.legend(handles=legend_elements1, loc=1, prop={'size': 20},  bbox_to_anchor=(0.9, 2.9), ncol=5)
axes.add_artist(legend1)
legend2 = axes.legend(handles=legend_elements2, loc=2, prop={'size': 20},  bbox_to_anchor=(-0.3, 2.65), ncol=2)
axes.add_artist(legend2)
# axes.axis("off")
# plt.tight_layout()
plt.subplots_adjust(left=0.07, bottom=0.08, top=0.75)

plt.suptitle("f (M$_p$|R$_p$, S, M$_*$)", fontsize=28, weight="bold")
plt.show(block=False)

plt.savefig(os.path.join(PlotFolder, '4d_InsolationPanel.pdf'), dpi=360)
# """
