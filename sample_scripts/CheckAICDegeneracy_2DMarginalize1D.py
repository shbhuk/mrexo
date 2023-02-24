import matplotlib.pyplot as plt
import matplotlib
import numpy as np
# from scipy.interpolate import UnivariateSpline

import glob, os
# import imageio

from mrexo.mle_utils_nd import calculate_conditional_distribution
from mrexo.aic_nd import MakePlot


matplotlib.rcParams['xtick.labelsize'] = 25
matplotlib.rcParams['ytick.labelsize'] = 25

################ Run Conditional Distribution ################ 
from mrexo.mle_utils_nd import calculate_conditional_distribution

# ConditionString = 'm|r,p'
ConditionString = 'm|r'
# ConditionString = 'm,r|p'

# Script to check AIC, and generate conditional distributions for all the models in the lowest contour in the AIC 2D plot
# Before running this, need to run CheckAICDegeneracy_Fit2D.py

AsymmResultDirectory = r"C:\Users\skanodia\Documents\GitHub\mrexo\sample_scripts\TestRuns\Trial_FGKM_2D_MR_aic_asymm_degmin10_20c"
AIC = np.load(os.path.join(AsymmResultDirectory, 'output', 'other_data_products', 'AIC.npy'))
DegreeCandidates = np.loadtxt(os.path.join(AsymmResultDirectory, 'output', 'other_data_products', 'degree_candidates.txt')).astype(int)

a=r"""
Fig_AIC = MakePlot(AIC, Title='AIC', degree_candidates=DegreeCandidates, Interpolate=True, AddContour=True)

Slope = np.sqrt(np.diff(AIC, axis=0, prepend=np.nan)**2 + np.diff(AIC, axis=1, prepend=np.nan)**2)
# Fig_d_AIC = MakePlot(Slope, Title='Derivative of AIC', degree_candidates=DegreeCandidates, Interpolate=False, AddContour=True)

AIC_Contour = plt.contour(DegreeCandidates[0], DegreeCandidates[1], AIC, 20); plt.close()
DegreeMesh = np.meshgrid(DegreeCandidates[0], DegreeCandidates[1]) 
DegreeX = DegreeMesh[0][AIC < AIC_Contour.levels[1]]
DegreeY = DegreeMesh[1][AIC < AIC_Contour.levels[1]]
print(DegreeX.shape)

# Comparing the posteriors for conditional distributions across a 2D grid of asymmetric degrees in MR space based on Asymmetric degree AIC run
PosteriorComparisonGridDirectory = r"C:\Users\skanodia\Documents\GitHub\mrexo\sample_scripts\TestRuns\FGKM_2D_MR"
# PosteriorComparisonGridRuns = os.listdir(PosteriorComparisonGridDirectory)
# degree_candidates =  np.array([np.linspace(10, d, 20, dtype=int) for d in [60, 60]])
# Titles = ['MR {}-{}'.format(x, y) for x in degree_candidates[0] for y in degree_candidates[0]]

PosteriorComparisonGridRuns = ['Trial_FGKM_2D_MR_{}_{}'.format(x, y) for x, y in zip(DegreeX, DegreeY)]
Titles = ['MR {}-{}'.format(x, y) for x, y in zip(DegreeX, DegreeY)]
"""
Runs = PosteriorComparisonGridRuns# 

# Runs = [RunName1]#, RunName2, RunName3]
# Titles = ['FGK 2022: #348', 'M dwarf 2019: #4', 'M dwarf 2022: #18']
TitlePos = np.repeat(350, len(Runs))

n = 17
# fig, ax = plt.subplots(n, sharex=True, sharey=True, figsize=(6,9))
fig, ax = plt.subplots(1, sharex=True, sharey=True, figsize=(6,9))
ax = [ax]

cmap = matplotlib.cm.Spectral
norm = matplotlib.colors.Normalize(vmin=0, vmax=17)
# color = cmap(norm(17))

for d, RunName in enumerate(Runs):

	save_path = os.path.join(r"C:\Users\skanodia\Documents\GitHub\mrexo\sample_scripts", 'TestRuns', 'FGKM_2D_MR', RunName)

	ConditionName = '2D_6Re_'+ConditionString.replace('|', '_').replace(',', '_')
	PlotFolder = os.path.join(save_path, ConditionName)

	deg_per_dim = np.loadtxt(os.path.join(save_path, 'output', 'deg_per_dim.txt'), dtype=int)
	DataDict = np.load(os.path.join(save_path, 'input', 'DataDict.npy'), allow_pickle=True).item()
	DataSequences = np.loadtxt(os.path.join(save_path, 'output', 'other_data_products', 'DataSequences.txt'))
	weights = np.loadtxt(os.path.join(save_path, 'output', 'weights.txt'))
	JointDist = np.load(os.path.join(save_path, 'output', 'JointDist.npy'), allow_pickle=True)

	Condition = ConditionString.split('|')
	LHSTerms = Condition[0].split(',')
	RHSTerms = Condition[1].split(',')
	deg_vec_per_dim = [np.arange(1, deg+1) for deg in deg_per_dim] 

	LHSDimensions = np.array([(np.arange(DataDict['ndim'])[np.isin(DataDict['ndim_char'] , l)])[0] for l in LHSTerms])
	RHSDimensions = np.array([(np.arange(DataDict['ndim'])[np.isin(DataDict['ndim_char'] , r)])[0] for r in RHSTerms])


	xseq = DataSequences[LHSDimensions[0]]
	yseq = DataSequences[RHSDimensions[0]]
	# t = outputs['DataSequence'][3]

	DataDict = DataDict
	MeasurementDict = {'r':[[1, 2], [np.nan, np.nan]], 'p':[[1, 1], [np.nan, np.nan]], 'stm':[[0.1, 0.3], [np.nan, np.nan]]}
	MeasurementDict = {'r':[[10**0.2, 10**0.4, 10**0.6], [np.nan, np.nan, np.nan]]}#, 'p':[[1, 1, 10], [np.nan, np.nan]], 'stm':[[0.5], [np.nan, np.nan]]}
	MeasurementDict = {'stm':[[0.2, 0.4, 0.43, 0.46, 0.49, 0.52, 0.55, 0.57, 0.6], [np.nan]*9]}#, 'r':[[1], [np.nan]]}
	MeasurementDict = {RHSTerms[0]:[[10**0.0], [np.nan]]}


	r = [13]
	colours = ["C2", "C1", "C0"]
	colours = [cmap(norm(d))]
	MeasurementDict = {'r':[r, np.repeat(np.nan, len(r))]}
	LogMeasurementDict = {
												'r':[np.log10(r),  np.reshape(np.repeat(np.nan, 2*len(r)), (len(r), 2))]
											}



	ConditionalDist, MeanPDF, VariancePDF = calculate_conditional_distribution(ConditionString, DataDict, weights, deg_per_dim,
		JointDist, LogMeasurementDict)
		
	# LinearVariancePDF = (10**MeanPDF * np.log(10))**2 * VariancePDF
	# LinearSigmaPDF = np.sqrt(LinearVariancePDF)


	d = 0
	# fig = plt.figure(figsize=(8.5,6.5))
	_ = [ax[d].plot(10**xseq, ConditionalDist[i], label='Radius = '+str(np.round(r[i], 1))+'$ R_{\oplus}$', c=colours[i]) for i in range(len(r))]
	_ = [ax[d].axvline(10**MeanPDF[i], linestyle='dashed', c=colours[i]) for i in range(len(r))]
	# _ = [ax[d].text(10**MeanPDF[i]*(1.1), 1.5, str(np.round(10**MeanPDF[i], 2)) + '$ M_{\oplus}$', fontsize=18, c=colours[i]) for i in range(len(r))]
	# plt.title(DataDict['ndim_label'][2]+" = {:.3f}".format(MeasurementDict[RHSTerms[0]][0][i]))

	# ax[d].text(TitlePos[d], 1.5, RunName[-8:], fontsize=22)

	YTicks = [0, 0.5, 1, 1.5]

	YLabels = np.round(YTicks, 2)


	# plt.xlim(10**DataDict['ndim_bounds'][0][0], 10**DataDict['ndim_bounds'][0][1])
	ax[d].set_xscale("log")
	ax[d].set_xlim(10, 2000)
	ax[d].set_ylim(0, 2.3)
	
	
# ax[-1].set_xlabel(DataDict['ndim_label'][RHSDimensions[0]])
# ax[5].set_ylabel("Probability Density Function")
# ax[0].set_xlabel(DataDict['ndim_label'][LHSDimensions[0]], size=25)
# ax[0].legend(loc=2, fontsize=15)
fig.subplots_adjust(hspace=0.01)
# plt.tight_layout()
ax[0].set_title("Comparing Samples")
plt.show(block=False)

