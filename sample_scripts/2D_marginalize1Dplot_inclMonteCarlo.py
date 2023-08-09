import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import glob, os

from mrexo.mle_utils_nd import calculate_conditional_distribution


matplotlib.rcParams['xtick.labelsize'] = 25
matplotlib.rcParams['ytick.labelsize'] = 25

################ Run Conditional Distribution ################ 
from mrexo.mle_utils_nd import calculate_conditional_distribution

# ConditionString = 'm|r,p'
ConditionString = 'mdust|stm'

UseMonteCarlo = False
Runs = ['AIC_4real_MonteCarlo_2D']#, 'AIC_4real_MonteCarlo_2D']
Runs = ['d20_MR_MC100_2D', 'd40_MR_MC100_2D', 'd80_MR_MC100_2D', 'd100_MR_MC100_2D']
Runs = ['AllPlanet_RpLt4_StMlt1.5_MR_CV_100MC_100BS']
Runs = ['LupusClassII_2d_CV_100MC_100BS']
SupTitle = '' # Testing Monte-Carlo'

Titles = np.repeat('', len(Runs))
# Titles = ["{}$\sigma$".format(str(np.round(1/s, 1))) for s in np.array(Sigma).astype(float)] #np.round(1/np.array(Sigma).astype(float), 2)
Titles = ['', '20x20', '40x40', '80x80', '100x100']
TitlePos = np.repeat(100, len(Runs))


fig, ax = plt.subplots(len(Runs), sharex=True, sharey=True, figsize=(6, 6))
if np.size(ax) == 1: ax=[ax]

for d, RunName in enumerate(Runs):
	save_path = os.path.join(r"C:\Users\skanodia\Documents\GitHub\mrexo\sample_scripts", 'TestRuns', RunName)
	# save_path = os.path.join(r"/storage/home/szk381/work/mrexo/sample_scripts", 'TestRuns', RunName)

	ConditionName = '2D_StMass_'+ConditionString.replace('|', '_').replace(',', '_')
	PlotFolder = os.path.join(save_path, ConditionName)
	MonteCarloFolder = os.path.join(save_path, 'output', 'other_data_products', 'MonteCarlo')
	# MonteCarloFolder = os.path.join(save_path, 'output', 'other_data_products', 'Bootstrap')

	UseMonteCarlo = os.path.exists(MonteCarloFolder)
	NumMonteCarlo =  len(glob.glob(os.path.join(MonteCarloFolder, 'JointDist_MCSim*.npy')))

	deg_per_dim = np.loadtxt(os.path.join(save_path, 'output', 'deg_per_dim.txt')).astype(int)
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


	r = [1, 1.5, 2, 3]
	r = [0.1, 0.3, 0.5, 1.0]
	colours = ["C3", "C2", "C1", "C0", "C4"]
	MeasurementDict = {'r':[r, np.repeat(np.nan, len(r))]}
	LogMeasurementDict = {
												'stm':[np.log10(r),  np.reshape(np.repeat(np.nan, 2*len(r)), (len(r), 2))]
											}

	ConditionalDist, MeanPDF, VariancePDF = calculate_conditional_distribution(ConditionString, DataDict, weights, deg_per_dim,
		JointDist, LogMeasurementDict)

	if UseMonteCarlo:
		ConditionalMC = np.zeros((NumMonteCarlo, *np.shape(ConditionalDist)))
		MeanMC = np.zeros((NumMonteCarlo, *np.shape(MeanPDF)))
		VarianceMC = np.zeros((NumMonteCarlo, *np.shape(VariancePDF)))
	
		print("Conditioning the model from each Monte-Carlo simulation")
		for mc in range(NumMonteCarlo):
				weights_mc = np.loadtxt(os.path.join(MonteCarloFolder, 'weights_MCSim{}.txt'.format(str(mc))))
				JointDist_mc = np.load(os.path.join(MonteCarloFolder, 'JointDist_MCSim{}.npy'.format(str(mc))), allow_pickle=True)
				# print(JointDist_mc.sum(), np.percentile(weights_mc, 95))
				ConditionalMC[mc], MeanMC[mc], VarianceMC[mc] = calculate_conditional_distribution(ConditionString, DataDict, weights_mc, deg_per_dim,
					JointDist_mc, LogMeasurementDict)


	# y = 10^x
	# dy = y * dx * ln(10)

	# LogSigmaPDF = np.sqrt(VariancePDF)
	# LinearSigmaPDF = 10**MeanPDF * np.log(10) * LogSigmaPDF
	# LinearVariancePDF = (10**MeanPDF * np.log(10))**2 * VariancePDF
	# LinearSigmaPDF = np.sqrt(LinearVariancePDF)

	# fig = plt.figure(figsize=(8.5,6.5))
	if UseMonteCarlo:
		# _ = [ax[d].fill_betweenx(y=np.arange(ax[d].get_ylim()[0], ax[d].get_ylim()[1]*5), x1=10**(), x2=10**(), alpha=0.2, color=colours[i]) for i in range(len(r))]
		print(1)
		# bins = 
		# logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
		for i in range(len(r)):
			hist = np.histogram(10**MeanMC[:,i])
			ax[d].hist(hist[1][:-1], hist[1], weights=hist[0]/np.max(hist[0]),  label='Stellar Mass = '+str(r[i])+' M$_{\odot}$', color=colours[i])
			
		# _ = [ax[d].hist(10**MeanMC[:,i], label='Radius = '+str(np.round(r[i], 1))+' R$_{\oplus}$', color=colours[i]) for i in range(len(r))]
	else:
		_ = [ax[d].fill_betweenx(y=np.arange(ax[d].get_ylim()[0], ax[d].get_ylim()[1]*5), x1=10**(MeanPDF[i] - np.sqrt(VariancePDF[i])), x2=10**(MeanPDF[i] + np.sqrt(VariancePDF[i])), alpha=0.2, color=colours[i]) for i in range(len(r))]
		_ = [ax[d].text(10**MeanPDF[i]*(1.1), 1, str(np.round(MeanPDF[i]/np.sqrt(VariancePDF[i]), 1)) + '$\sigma$', fontsize=22, c=colours[i]) for i in range(len(r))]	

	# _ = [ax[d].plot(10**xseq, ConditionalDist[i], label='Radius = '+str(np.round(r[i], 1))+' R$_{\oplus}$', c=colours[i]) for i in range(len(r))]


	_ = [ax[d].axvline(10**MeanPDF[i], linestyle='dashed', c=colours[i]) for i in range(len(r))]
	_ = [ax[d].text(10**MeanPDF[i]*(1.05), 1.2, str(np.round(10**MeanPDF[i], 1)) + ' M$_{\oplus}$', fontsize=22, c=colours[i]) for i in range(len(r))]

	# _ = [ax[d].axvline(10**xseq[np.argmax(ConditionalDist[i])], linestyle='solid', c=colours[i]) for i in range(len(r))]



	# plt.title(DataDict['ndim_label'][2]+" = {:.3f}".format(MeasurementDict[RHSTerms[0]][0][i]))

	ax[d].text(TitlePos[d], 1, Titles[d], fontsize=22)

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
	ax[d].set_xlim(1, 200)
	ax[d].set_ylim(0, 2)
	ax[0].set_ylabel("Probability Density Function")

ax[-1].set_xlabel(DataDict['ndim_label'][LHSDimensions[0]], size=25)
ax[0].legend(loc=0, fontsize=15)
fig.subplots_adjust(hspace=0.01)
# plt.tight_layout()
ax[0].set_title(SupTitle)
plt.show(block=False)


r"""
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
"""
