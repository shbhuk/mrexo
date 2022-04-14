import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import glob, os
import pandas as pd
# import imageio

from scipy.interpolate import UnivariateSpline
from mrexo.mle_utils_nd import calculate_conditional_distribution

ConditionString = 'm|r,stm'

################ Run Conditional Distribution ################ 
from mrexo.mle_utils_nd import calculate_conditional_distribution, NumericalIntegrate2D

# ConditionString = 'm|r,p'
# ConditionString = 'm|r'
# ConditionString = 'm,r|p'
# ConditionString = 'r,p|stm'
# ConditionString = 'm|r,stm'
# ConditionString = 'm,r|p,stm'
# ConditionString = 'm,r|feh'
# ConditionString = 'm,r|p'
# ConditionString = 'r|stm'


RunName = "Mdwarf_3D_deg45_20220409_M_R_StM1.2_v2_bounded"
# RunName = "Fake_3D_MRS"
save_path = os.path.join(r"C:\Users\shbhu\Documents\GitHub\mrexo\sample_scripts", 'TestRuns', 'ThesisRuns', RunName)

deg_per_dim = np.loadtxt(os.path.join(save_path, 'output', 'deg_per_dim.txt'), dtype=int)
DataDict = np.load(os.path.join(save_path, 'input', 'DataDict.npy'), allow_pickle=True).item()
DataSequences = np.loadtxt(os.path.join(save_path, 'output', 'other_data_products', 'DataSequences.txt'))
weights = np.loadtxt(os.path.join(save_path, 'output', 'weights.txt'))
JointDist = np.load(os.path.join(save_path, 'output', 'JointDist.npy'), allow_pickle=True).T # This transpose is required based on testing on simulated data Fake_3D_MRS

ConditionName = '3D_GG_'+ConditionString.replace('|', '_').replace(',', '_')
PlotFolder = os.path.join(save_path, ConditionName)

Condition = ConditionString.split('|')
LHSTerms = Condition[0].split(',')
RHSTerms = Condition[1].split(',')
deg_vec_per_dim = [np.arange(1, deg+1) for deg in deg_per_dim] 


LHSDimensions = np.array([(np.arange(DataDict['ndim'])[np.isin(DataDict['ndim_char'] , l)])[0] for l in LHSTerms])
RHSDimensions = np.array([(np.arange(DataDict['ndim'])[np.isin(DataDict['ndim_char'] , r)])[0] for r in RHSTerms])


xseq = DataSequences[LHSDimensions[0]]
yseq = DataSequences[RHSDimensions[0]]
zseq = DataSequences[RHSDimensions[1]]
# t = outputs['DataSequence'][3]

DataDict = DataDict
MeasurementDict = {'r':[[1, 2], [np.nan, np.nan]], 'p':[[1, 1], [np.nan, np.nan]], 'stm':[[0.1, 0.3], [np.nan, np.nan]]}
MeasurementDict = {'r':[[10**0.2, 10**0.4, 10**0.6], [np.nan, np.nan, np.nan]]}#, 'p':[[1, 1, 10], [np.nan, np.nan]], 'stm':[[0.5], [np.nan, np.nan]]}
MeasurementDict = {'stm':[[0.55, 0.6, 0.75, 1.0], [np.nan]*4], 'r':[[12], [np.nan]]}
# MeasurementDict = {'st':[[1.05], [np.nan]], 'r':[[12], [np.nan]]}
# MeasurementDict = {'r':[[10], [np.nan]], 'm':[[100], [np.nan]]}

# MeasurementDict = {'r':[[1, 1, 1], [np.nan, np.nan, np.nan]], 'p':[[1, 5, 10], [np.nan, np.nan, np.nan]]}
# MeasurementDict = {RHSTerms[0]:[[10**0.0], [np.nan]]}
# MeasurementDict = {'r':[[1], [np.nan]]}

# LogMeasurementDict = {ke:np.log10(MeasurementDict[ke]) for ke in MeasurementDict.keys()}

fig, ax = plt.subplots()#1, sharex=True, sharey=True, figsize=(1,6.5))

cmap =matplotlib.cm.Spectral
# Establish colour range based on variable
norm = matplotlib.colors.Normalize(vmin=0.5, vmax=1.15)



for j in range(len(MeasurementDict[RHSTerms[0]][0])):
	for k in range(len(MeasurementDict[RHSTerms[1]][0])):
		# print(j, k)
		
		colour = cmap(norm(MeasurementDict[RHSTerms[1]][0][k]))
		LogMeasurementDict = {RHSTerms[0]:[[np.log10(MeasurementDict[RHSTerms[0]][0][j])], [np.nan]], RHSTerms[1]:[[np.log10(MeasurementDict[RHSTerms[1]][0][k])], [np.nan]]}
	
		ConditionalDist, MeanPDF, VariancePDF = calculate_conditional_distribution(ConditionString, DataDict, weights, deg_per_dim,
			JointDist, LogMeasurementDict)
	
		ConditionPDF = UnivariateSpline(xseq, ConditionalDist[0]).integral(
			DataDict["ndim_bounds"][LHSDimensions[0]][0], DataDict["ndim_bounds"][LHSDimensions[0]][1])
			
		ax.plot(10**xseq, ConditionalDist[0]/ConditionPDF, label=r'St. Mass = {:.2f} '.format(MeasurementDict[RHSTerms[1]][0][k])+' M$_{\odot}$', c=colour)
		ax.axvline(10**MeanPDF[0], linestyle='dashed', c=colour)


XTicks = [10, 30, 100, 300, 1000]
# plt.ylim(0, 20)
ax.set_xticks(XTicks)
ax.set_xticklabels(["{:1d}".format(s) for s in XTicks])
print(MeasurementDict)
plt.xlabel("Planet Mass ($M_{\oplus}$)")
plt.ylabel("Probability Density Function")
plt.legend()
plt.title("Expected Planetary Mass for 12 Re vs Host Star Mass\n f(pl_masse|pl_rade=12, stm)", fontsize=15)
plt.tight_layout()
plt.grid(alpha=0.3)
plt.show(block=False)

"""
# for j in 
for k in np.arange(0, len(zseq), 2, dtype=int):
	
	ChosenZ = zseq[k]
	print(10**ChosenZ)

	MeasurementDict = {RHSTerms[0]:[[10**ChosenZ], [np.nan]]}
	LogMeasurementDict = {ke:np.log10(MeasurementDict[ke]) for ke in MeasurementDict.keys()}



	ConditionalDist, MeanPDF, VariancePDF = calculate_conditional_distribution(ConditionString, DataDict, weights, deg_per_dim,
		JointDist, LogMeasurementDict)


	_ = NumericalIntegrate2D(xseq, yseq, ConditionalDist[0], [xseq.min(), xseq.max()], [yseq.min(), yseq.max()])
	print(_)

	i=0
	fig = plt.figure(figsize=(8.5,6.5))
	im = plt.imshow(ConditionalDist[i], extent=(xseq.min(), xseq.max(), yseq.min(), yseq.max()), aspect='auto', origin='lower'); 
	# plt.plot(np.log10(Mass), np.log10(Radius),  'k.')
	plt.title(DataDict['ndim_label'][2]+" = {:.3f}".format(MeasurementDict[RHSTerms[0]][0][i]))
	plt.xlabel(DataDict['ndim_label'][LHSDimensions[0]])
	plt.ylabel(DataDict['ndim_label'][LHSDimensions[1]])

	plt.xlim(DataDict['ndim_bounds'][0][0], DataDict['ndim_bounds'][0][1])
	plt.ylim(DataDict['ndim_bounds'][1][0], DataDict['ndim_bounds'][1][1])
	plt.tight_layout()

	XTicks = np.linspace(xseq.min(), xseq.max(), 5)
	# XTicks = np.log10(np.array([0.3, 1, 3, 10, 30, 100, 300]))
	# XTicks = np.log10(np.array([1, 2, 3, 5, 10]))
	XTicks = np.log10(np.array([1.0, 1.5, 2, 2.5, 3, 4]))
	YTicks = np.linspace(yseq.min(), yseq.max(), 5)
	YTicks = np.log10(np.array([0.5, 1, 10, 30,  50, 100, 300]))
	# YTicks = np.log10(np.array([1, 5, 10, 30, 50]))

	XLabels = np.round(10**XTicks, 1)
	YLabels = np.round(10**YTicks, 2)

	plt.xticks(XTicks, XLabels)
	plt.yticks(YTicks, YLabels)
	cbar = fig.colorbar(im, ticks=[np.min(ConditionalDist[i]), np.max(ConditionalDist[i])], fraction=0.037, pad=0.04)
	cbar.ax.set_yticklabels(['Min', 'Max'])
	plt.tight_layout()
	# plt.show(block=False)


	plt.savefig(os.path.join(PlotFolder, ConditionName+'_z_{}.png'.format(np.round(MeasurementDict[RHSTerms[0]][0][i],3))))
	plt.close("all")
"""
	
###############
"""
ListofPlots = glob.glob(os.path.join(PlotFolder, '3*.png'))
ListofPlots.sort(key=os.path.getmtime)

writer = imageio.get_writer(os.path.join(PlotFolder, ConditionName+'.mp4'), fps=2)

for im in ListofPlots:
		#print(order, im)
		writer.append_data(imageio.imread(os.path.join(PlotFolder, im)))
writer.close()
"""
