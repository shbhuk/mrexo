import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import glob, os

from mrexo.mle_utils_nd import calculate_conditional_distribution

matplotlib.rcParams['xtick.labelsize'] = 25
matplotlib.rcParams['ytick.labelsize'] = 25
cmap = matplotlib.cm.viridis
cmap = matplotlib.cm.Spectral


################ Run Conditional Distribution ################ 

# Specify condition string based on the characters defined in DataDict for different dimensions. 
# This particular example is for a 2D fit.
ConditionString = 'm|r'

UseMonteCarlo = False
UseBootstrap = False

# Directory name for each run to compare. Can specify more than one if want to compare multiple datasets or fits
Runs = ['AllPlanet_RpLt4_StMlt1.5_MR_CV_100MC_100BS']

# Specify the master title for the figure
SuperTitle = '' # Testing Monte-Carlo'

# Specify the title for each run that is being compared
Titles = ['Run 1', 'Run 2']
Titles = np.repeat('', len(Runs))
TitlePos = np.repeat(100, len(Runs))


fig, ax = plt.subplots(len(Runs), sharex=True, sharey=True, figsize=(9, 6))
if np.size(ax) == 1: ax=[ax]

for d, RunName in enumerate(Runs):
	# Specify the path for the result directories
	save_path = os.path.join(r"C:\Users\skanodia\Documents\GitHub\mrexo\sample_scripts", 'TestRuns', RunName)

	ConditionName = '2D_'+ConditionString.replace('|', '_').replace(',', '_')
	PlotFolder = os.path.join(save_path, ConditionName) # Folder to save plots if needed
	if not os.path.exists(PlotFolder): os.makedirs(PlotFolder)

	if UseMonteCarlo:
		MonteCarloFolder = os.path.join(save_path, 'output', 'other_data_products', 'MonteCarlo')
		if not os.path.exists(MonteCarloFolder): raise Exception("Directory with Monte-Carlo results does not exist. Set `UseMonteCarlo` = False")
		NumMonteCarlo =  len(glob.glob(os.path.join(MonteCarloFolder, 'JointDist_MCSim*.npy')))

	if UseBootstrap:
		BootstrapFolder = os.path.join(save_path, 'output', 'other_data_products', 'Bootstrap')
		if not os.path.exists(BootstrapFolder): raise Exception("Directory with Bootstrap results does not exist. Set `UseBootstrap` = False")
		NumBootstrap =  len(glob.glob(os.path.join(MonteCarloFolder, 'JointDist_MCSim*.npy')))

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


	r = [1, 2,  3, 4]
	# colours = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]

	norm = matplotlib.colors.Normalize(vmin=np.min(r), vmax=np.max(r))
	colours = [cmap(norm(r[i])) for i in range(len(r))]


	# The keyword for the measurement dictionary must be the RHS of the condition string.
	# Final measurement dictionary must be in log10 units.
	# Can specify measurement uncertainty if needed but np.nan works fine for the most bit.

	LogMeasurementDict = {
												'r':[np.log10(r),  [[np.nan, np.nan]]*len(r)]
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
				ConditionalMC[mc], MeanMC[mc], VarianceMC[mc] = calculate_conditional_distribution(ConditionString, DataDict, weights_mc, deg_per_dim,
					JointDist_mc, LogMeasurementDict)

	if UseBootstrap:
		ConditionalBS = np.zeros((NumBootstrap, *np.shape(ConditionalDist)))
		MeanBS = np.zeros((NumBootstrap, *np.shape(MeanPDF)))
		VarianceBS = np.zeros((NumBootstrap, *np.shape(VariancePDF)))
	
		print("Conditioning the model from each Bootstrap simulation")
		for bs in range(NumBootstrap):
				weights_bs = np.loadtxt(os.path.join(BootstrapFolder, 'weights_BSSim{}.txt'.format(str(bs))))
				JointDist_bs = np.load(os.path.join(BootstrapFolder, 'JointDist_BSSim{}.npy'.format(str(bs))), allow_pickle=True)
				ConditionalBS[bs], MeanBS[bs], VarianceBS[bs] = calculate_conditional_distribution(ConditionString, DataDict, weights_bs, deg_per_dim,
					JointDist_bs, LogMeasurementDict)




	if UseMonteCarlo:
		for i in range(len(r)):
			hist = np.histogram(10**MeanMC[:,i])
			if i == 0: ax[d].hist(hist[1][:-1], hist[1], weights=hist[0]/np.max(hist[0]),  label='Monte-Carlo', color=colours[i], alpha=0.5)
			else: ax[d].hist(hist[1][:-1], hist[1], weights=hist[0]/np.max(hist[0]),  color=colours[i], alpha=0.5)
	if UseBootstrap:
		for i in range(len(r)):
			hist = np.histogram(10**MeanBS[:,i])
			if i ==0: ax[d].hist(hist[1][:-1], hist[1], weights=hist[0]/np.max(hist[0]),  label='Bootstrap', color=colours[i], alpha=1.0)	
			else: ax[d].hist(hist[1][:-1], hist[1], weights=hist[0]/np.max(hist[0]),  color=colours[i], alpha=1.0)	

	# if not (UseMonteCarlo | UseBootstrap):
		# _ = [ax[d].fill_betweenx(y=np.arange(ax[d].get_ylim()[0], ax[d].get_ylim()[1]*5), x1=10**(MeanPDF[i] - np.sqrt(VariancePDF[i])), x2=10**(MeanPDF[i] + np.sqrt(VariancePDF[i])), alpha=0.2, color=colours[i]) for i in range(len(r))]
		# _ = [ax[d].text(10**MeanPDF[i]*(1.1), 1, str(np.round(MeanPDF[i]/np.sqrt(VariancePDF[i]), 1)) + '$\sigma$', fontsize=22, c=colours[i]) for i in range(len(r))]	
	_ = [ax[d].plot(10**xseq, ConditionalDist[i]/np.nanmax(ConditionalDist[i]), c=colours[i], label=str(r[i]) + ' ' + DataDict['ndim_label'][RHSDimensions[0]]) for i in range(len(r))]
	
	_ = [ax[d].axvline(10**MeanPDF[i], linestyle='dashed', c=colours[i]) for i in range(len(r))]
	_ = [ax[d].text(10**MeanPDF[i]*(1.05), 0.8, str(np.round(10**MeanPDF[i], 1)) + ' ' + DataDict['ndim_label'][LHSDimensions[0]], fontsize=22, c=colours[i]) for i in range(len(r))]
	ax[d].text(TitlePos[d], 1, Titles[d], fontsize=22)



	# plt.title(DataDict['ndim_label'][2]+" = {:.3f}".format(MeasurementDict[RHSTerms[0]][0][i]))
	# XTicks = np.linspace(xseq.min(), xseq.max(), 5)
	# XLabels = np.round(10**XTicks, 1)
	# ax[d].set_xticks(10**XTicks)
	# ax[d].set_xticklabels(XLabels)
	# ax[d].set_xscale("log")

	# ax[d].set_ylim(0, 2)
	ax[0].set_ylabel("Probability Density Function")

ax[-1].set_xlabel(DataDict['ndim_label'][LHSDimensions[0]], size=25)
ax[0].legend(loc=0, fontsize=15)
fig.subplots_adjust(hspace=0.01)
ax[0].set_title(SuperTitle)
plt.show(block=False)

# Can then save the plot in PlotFolder if required
# plt.savefig(os.path.join(PlotFolder, 'SomeFigureNameHere.pdf'))

# Outputs in log10
# Standard Fit : ConditionalDist, MeanPDF, VariancePDF 
# Monte-Carlo : ConditionalMC, MeanMC, VarianceMC  
# Bootstrap : ConditionalBS, MeanBS, VarianceBS
