import matplotlib.pyplot as plt
import matplotlib
import numpy as np
# from scipy.interpolate import UnivariateSpline

import glob, os
# import imageio

from mrexo.mle_utils_nd import calculate_conditional_distribution


matplotlib.rcParams['xtick.labelsize'] = 25
matplotlib.rcParams['ytick.labelsize'] = 25

################ Run Conditional Distribution ################ 
from mrexo.mle_utils_nd import calculate_conditional_distribution

# ConditionString = 'm|r,p'
ConditionString = 'm|r'
# ConditionString = 'm,r|p'
# ConditionString = 'r,p|stm'
# ConditionString = 'm|r,stm'
# ConditionString = 'm,r|p,stm'
# ConditionString = 'm,r|feh'
# ConditionString = 'm,r|p'
# ConditionString = 'r|stm'

RunName1 = 'Kepler_127_M_R'; d=0 # Original Ning 2018 Kepler sample with 127 planets
RunName2= 'Mdwaf_24_Kanodia2019_M_R_deg17'; d=1 # Original Kanodia 2019 M dwarf sample with 24 planets
RunName3 = 'Mdwarf_2D_20220325_M_R'; d=2 # Sample from 20220325 w/ 63 M dwarf planets

RunName1 = 'Kepler_127_M_R_bounded'; d=0 # Original Ning 2018 Kepler sample with 127 planets
RunName2= 'Mdwarf_Kanodia2019_bounded'; d=1 # Original Kanodia 2019 M dwarf sample with 24 planets
RunName3 = 'Mdwarf_2D_20220325_M_R_bounded'; d=2 # Sample from 20220325 w/ 63 M dwarf planets

Runs = [RunName1, RunName2, RunName3]
Titles = ['Kepler (FGK) 2018: #127', 'M dwarf 2019: #24', 'M dwarf 2022: #63']
TitlePos = [130, 270, 270]
fig, ax = plt.subplots(3, sharex=True, sharey=True, figsize=(15,6.5))


for d, RunName in enumerate(Runs):

	save_path = os.path.join(r"C:\Users\shbhu\Documents\GitHub\mrexo\sample_scripts", 'TestRuns', 'ThesisRuns', RunName)



	ConditionName = '2D_1Re_'+ConditionString.replace('|', '_').replace(',', '_')
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
	# MeasurementDict = {'r':[[1, 1, 1], [np.nan, np.nan, np.nan]], 'p':[[1, 5, 10], [np.nan, np.nan, np.nan]]}
	MeasurementDict = {RHSTerms[0]:[[10**0.0], [np.nan]]}


	r = [1.67, 3, 12]
	colours = ["C0", "C1", "C2"]
	MeasurementDict = {'r':[r, [np.nan, np.nan, np.nan]]}
	LogMeasurementDict = {ke:np.log10(MeasurementDict[ke]) for ke in MeasurementDict.keys()}



	ConditionalDist, MeanPDF, VariancePDF = calculate_conditional_distribution(ConditionString, DataDict, weights, deg_per_dim,
		JointDist, LogMeasurementDict)
		
	# LinearVariancePDF = (10**MeanPDF * np.log(10))**2 * VariancePDF
	# LinearSigmaPDF = np.sqrt(LinearVariancePDF)



	# fig = plt.figure(figsize=(8.5,6.5))
	_ = [ax[d].plot(10**xseq, ConditionalDist[i], label='Radius = '+str(np.round(r[i], 1))+'$ R_{\oplus}$', c=colours[i]) for i in range(len(r))]
	_ = [ax[d].axvline(10**MeanPDF[i], linestyle='dashed', c=colours[i]) for i in range(len(r))]
	_ = [ax[d].text(10**MeanPDF[i]*(1.1), 1.5, str(np.round(10**MeanPDF[i], 2)) + '$ M_{\oplus}$', fontsize=18, c=colours[i]) for i in range(len(r))]
	# plt.title(DataDict['ndim_label'][2]+" = {:.3f}".format(MeasurementDict[RHSTerms[0]][0][i]))

	ax[d].text(TitlePos[d], 1.9, Titles[d], fontsize=22)

	# XTicks = np.linspace(xseq.min(), xseq.max(), 5)
	# XTicks = np.log10(np.array([0.3, 1, 3, 10, 30, 100, 300]))
	# XTicks = np.log10(np.array([1, 2, 3, 5, 10]))
	# XTicks = np.log10(np.array([1.0, 1.5, 2, 2.5, 3, 4]))
	# YTicks = np.linspace(yseq.min(), yseq.max(), 5)
	# YTicks = np.log10(np.array([0.5, 1, 10, 30,  50, 100, 300]))
	YTicks = [0, 0.5, 1, 1.5]

	# XLabels = np.round(10**XTicks, 1)
	YLabels = np.round(YTicks, 2)
	# ax[d].set_yticks(YTicks)
	# ax[d].set_yticklabels(YLabels)



	# plt.xlim(10**DataDict['ndim_bounds'][0][0], 10**DataDict['ndim_bounds'][0][1])
	ax[d].set_xscale("log")
	ax[d].set_xlim(0.1, 2000)
	ax[d].set_ylim(0, 2.3)
	ax[1].set_ylabel("Probability Density Function")

ax[2].set_xlabel(DataDict['ndim_label'][LHSDimensions[0]], size=25)
ax[0].legend(loc=2, fontsize=15)
fig.subplots_adjust(hspace=0.01)
# plt.tight_layout()
ax[0].set_title("Comparing Samples")
plt.show(block=False)


# """
## Plot Joint Distribution 


for d, RunName in enumerate(Runs):

	save_path = os.path.join(r"C:\Users\shbhu\Documents\GitHub\mrexo\sample_scripts", 'TestRuns', 'ThesisRuns', RunName)



	ConditionName = '2D_1Re_'+ConditionString.replace('|', '_').replace(',', '_')
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
	plt.ylim(np.log10(0.5), DataDict['ndim_bounds'][1][1])
	plt.tight_layout()

	XTicks = [0.5,1, 3, 10, 20]
	YTicks = [0.5, 1, 3, 10, 30, 100, 300, 1000]
	
	XLabels = XTicks
	YLabels = YTicks

	plt.xticks(np.log10(XTicks), XLabels)
	plt.yticks(np.log10(YTicks), YLabels)

	cbar = fig.colorbar(im, ticks=[np.min(JointDist), np.max(JointDist)], fraction=0.037, pad=0.04)
	cbar.ax.set_yticklabels(['Min', 'Max'])
	plt.tight_layout()
	plt.savefig(os.path.join(r"C:\Users\shbhu\Documents\GitHub\mrexo\sample_scripts", 'TestRuns', 'ThesisRuns', RunName+'_JointDist.png'))
	plt.close("all")
# """
