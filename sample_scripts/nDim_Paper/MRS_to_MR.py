import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib import gridspec
from scipy.interpolate import RectBivariateSpline, UnivariateSpline
from scipy.integrate import simps

import numpy as np
# import imageio
import glob, os
import pandas as pd
from mrexo.mle_utils_nd import calculate_conditional_distribution, NumericalIntegrate2D
import datetime
# import gc # import garbage collector interface

# from pyastrotools.general_tools import SigmaAntiLog10, SigmaLog10
# from pyastrotools.astro_tools import calculate_orbperiod


ConditionString = 'm,insol|r'
ConditionName = '3D_'+ConditionString.replace('|', '_').replace(',', '_')

ResultDirectory = r'C:\Users\skanodia\Documents\GitHub\mrexo\sample_scripts\TestRuns\AllPlanet_RpLt4_StMlt1.5_MRS_d100_100MC_100BS'
PlotFolder = os.path.join(ResultDirectory, ConditionName)
PlotFolder = os.path.join(ResultDirectory, '3D_m_r_s')
MonteCarloFolder = os.path.join(ResultDirectory, 'output', 'other_data_products', 'MonteCarlo')


if not os.path.exists(PlotFolder):
	print("3D Plot folder does not exist")
	os.mkdir(PlotFolder)


DataDict = np.load(os.path.join(ResultDirectory, 'input', 'DataDict.npy'), allow_pickle=True).item()
JointDist = np.load(os.path.join(ResultDirectory, 'output', 'JointDist.npy'), allow_pickle=True)
weights = np.genfromtxt(os.path.join(ResultDirectory, 'output', 'weights.txt'))
deg_per_dim = np.loadtxt(os.path.join(ResultDirectory, 'output', 'deg_per_dim.txt')).astype(int)

Condition = ConditionString.split('|')
LHSTerms = Condition[0].split(',')
RHSTerms = Condition[1].split(',')
deg_vec_per_dim = [np.arange(1, deg+1) for deg in deg_per_dim] 

LHSDimensions = np.array([(np.arange(DataDict['ndim'])[np.isin(DataDict['ndim_char'] , l)])[0] for l in LHSTerms])
RHSDimensions = np.array([(np.arange(DataDict['ndim'])[np.isin(DataDict['ndim_char'] , r)])[0] for r in RHSTerms])

xseq = DataDict['DataSequence'][LHSDimensions[0]] # m
yseq = DataDict['DataSequence'][LHSDimensions[1]] # stm
zseq = DataDict['DataSequence'][RHSDimensions[0]]

UseMonteCarlo = True
NumMonteCarlo = 100

PlanetRadiusArray = [1.5]

MeasurementDict=  {
	'r':[
			[np.log10(PlanetRadiusArray)], [[np.nan, np.nan], [np.nan, np.nan]]
		]
	}
ConditionString = 'm,insol|r'

ConditionalDist, MeanPDF, VariancePDF = calculate_conditional_distribution(ConditionString, DataDict, weights, deg_per_dim,
	JointDist, MeasurementDict)
IntCDF = simps(ConditionalDist[0], yseq)
IntMean = simps(IntCDF*xseq, xseq)

plt.figure(figsize=(9,8))
# plt.axvline(10**MeanPDF[0,0], linestyle='dashed', color='k')
plt.axvline(10**IntMean, linestyle='dashed', color='k',  linewidth=2, zorder=100, label="2D : f(m|r = {}".format(PlanetRadiusArray[0])+r"$R_{\oplus}$)")
# plt.plot(10**xseq, IntCDF/np.max(IntCDF), c='k', linewidth=2, zorder=100, label="2D : f(m|r = {}".format(PlanetRadiusArray[0])+r"$R_{\oplus}$ )")
# plt.legend()
plt.show(block=False)



if UseMonteCarlo:
	ConditionalMC = np.zeros((NumMonteCarlo, *np.shape(ConditionalDist)))
	IntMeanMC = np.zeros((NumMonteCarlo, *np.shape(MeanPDF)))
	VarianceMC = np.zeros((NumMonteCarlo, *np.shape(VariancePDF)))
	print("Conditioning the model from each Monte-Carlo simulation")
	for mc in range(NumMonteCarlo):
			weights_mc = np.loadtxt(os.path.join(MonteCarloFolder, 'weights_MCSim{}.txt'.format(str(mc))))
			JointDist_mc = np.load(os.path.join(MonteCarloFolder, 'JointDist_MCSim{}.npy'.format(str(mc))), allow_pickle=True)
			# print(JointDist_mc.sum(), np.percentile(weights_mc, 95))
			ConditionalMC[mc], MeanMC, VarianceMC[mc] = calculate_conditional_distribution(ConditionString, DataDict, weights_mc, deg_per_dim,
				JointDist_mc, MeasurementDict)
			IntMeanMC[mc] = simps(simps(ConditionalMC[mc][0], yseq)*xseq, xseq)
			print(mc)
			# print(10**IntMeanMC[mc])

	hist = np.histogram(10**IntMeanMC[:,0,0])
	plt.hist(hist[1][:-1], hist[1], weights=hist[0]/np.max(hist[0]), color='grey', label="2D : f(m|r = {}".format(PlanetRadiusArray[0])+r"$R_{\oplus}$) MC")
	# plt.hist(10**IntMeanMC[:,0,0], density=True)



# InsolationArray = np.logspace(-0.5, 3, 20)
InsolationArray = [1, 3, 10, 30, 100, 300, 1000]

norm = matplotlib.colors.Normalize(vmin=np.log10(InsolationArray[0]), vmax=np.log10(InsolationArray[-1]))
cmap = matplotlib.cm.Spectral

CombinedQuery = np.rollaxis(np.array(np.meshgrid(InsolationArray, PlanetRadiusArray)), 0, 3).reshape(len(InsolationArray)*len(PlanetRadiusArray),2)

MeasurementDict=  {
			'insol':[np.log10(CombinedQuery[:,0]), [[np.nan, np.nan]]*len(CombinedQuery)], 
			'r':[np.log10(CombinedQuery[:,1]), [[np.nan, np.nan]]*len(CombinedQuery)]
}
ConditionString = 'm|r,insol'


ConditionalDist, MeanPDF, VariancePDF = calculate_conditional_distribution(ConditionString, DataDict, weights, deg_per_dim,
	JointDist.T, MeasurementDict)


# plt.figure()
for i,insol in enumerate(InsolationArray):
	if i%1 == 0: 	plt.axvline(10**MeanPDF[i], linestyle='dashed', color=cmap(norm(np.log10(insol))), label=r"3D f(m|r, {}".format(np.round(insol)) + r" $S_{\oplus}$)")
	else: plt.axvline(10**MeanPDF[i], linestyle='dashed', color=cmap(norm(np.log10(insol))))

		# plt.plot(10**xseq, ConditionalDist[i]/np.max(ConditionalDist[i]), color=cmap(norm(stm)), label=r"{:.2f}".format(stm) + r" $M_{\odot}$")
	# else: 	
		# plt.plot(10**xseq, ConditionalDist[i]/np.max(ConditionalDist[i]), color=cmap(norm(stm)))
	# plt.axvline(10**MeanPDF[i], linestyle='dashed', color=cmap(norm(stm)))


plt.legend(fontsize=20)
plt.xlabel(r"Planetary Mass ($M_{\oplus}$)", fontsize=30)
plt.ylabel("PDF", fontsize=30)
plt.title(r"f(m|r=1.5 $R_{\oplus}$, insol)")
plt.yticks(fontsize=35)
plt.xticks(fontsize=35)
plt.xlim(2, 9)
plt.tight_layout()
plt.show(block=False)

plt.savefig(os.path.join(r"C:\Users\skanodia\Documents\GitHub\Paper_ndim_mrexo\Plots", "MRS_MR_Rp1.5_v2.pdf"))
