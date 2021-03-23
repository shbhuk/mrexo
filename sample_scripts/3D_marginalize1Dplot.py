import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import glob
import imageio

from mrexo.mle_utils_nd import calculate_conditional_distribution

ConditionString = 'r,m|stm'

################ Run Conditional Distribution ################ 
from mrexo.mle_utils_nd import calculate_conditional_distribution

ConditionString = 'm|r,p'
# ConditionString = 'm|r'
# ConditionString = 'm,r|p'
ConditionString = 'r,p|stm'
# ConditionString = 'm|r,stm'
# ConditionString = 'm,r|p,stm'
# ConditionString = 'm,r|feh'
# ConditionString = 'm,r|p'
# ConditionString = 'r|stm'


ConditionName = '3D_URad_3Re_'+ConditionString.replace('|', '_').replace(',', '_')
PlotFolder = os.path.join(save_path, ConditionName)

Condition = ConditionString.split('|')
LHSTerms = Condition[0].split(',')
RHSTerms = Condition[1].split(',')
deg_vec_per_dim = [np.arange(1, deg+1) for deg in deg_per_dim] 


LHSDimensions = np.array([(np.arange(DataDict['ndim'])[np.isin(DataDict['ndim_char'] , l)])[0] for l in LHSTerms])
RHSDimensions = np.array([(np.arange(DataDict['ndim'])[np.isin(DataDict['ndim_char'] , r)])[0] for r in RHSTerms])


xseq = outputs['DataSequence'][LHSDimensions[0]]
yseq = outputs['DataSequence'][LHSDimensions[1]]
zseq = outputs['DataSequence'][2]
# t = outputs['DataSequence'][3]

DataDict = DataDict
MeasurementDict = {'r':[[1, 2], [np.nan, np.nan]], 'p':[[1, 1], [np.nan, np.nan]], 'stm':[[0.1, 0.3], [np.nan, np.nan]]}
MeasurementDict = {'r':[[10**0.2, 10**0.4, 10**0.6], [np.nan, np.nan, np.nan]]}#, 'p':[[1, 1, 10], [np.nan, np.nan]], 'stm':[[0.5], [np.nan, np.nan]]}
MeasurementDict = {'stm':[[0.2, 0.4, 0.43, 0.46, 0.49, 0.52, 0.55, 0.57, 0.6], [np.nan]*9]}#, 'r':[[1], [np.nan]]}
# MeasurementDict = {'r':[[1, 1, 1], [np.nan, np.nan, np.nan]], 'p':[[1, 5, 10], [np.nan, np.nan, np.nan]]}
MeasurementDict = {RHSTerms[0]:[[10**0.0], [np.nan]]}
# MeasurementDict = {'r':[[1], [np.nan]]}


for k in np.arange(0, len(zseq), 1, dtype=int):
	
	ChosenZ = zseq[k]
	print(10**ChosenZ)

	MeasurementDict = {RHSTerms[0]:[[10**ChosenZ], [np.nan]]}
	LogMeasurementDict = {ke:np.log10(MeasurementDict[ke]) for ke in MeasurementDict.keys()}



	ConditionalDist, MeanPDF, VariancePDF = calculate_conditional_distribution(ConditionString, DataDict, weights, deg_per_dim,
		JointDist, LogMeasurementDict)


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
	
	
###############

ListofPlots = glob.glob(os.path.join(PlotFolder, '3*.png'))
ListofPlots.sort(key=os.path.getmtime)

writer = imageio.get_writer(os.path.join(PlotFolder, ConditionName+'.mp4'), fps=2)

for im in ListofPlots:
		#print(order, im)
		writer.append_data(imageio.imread(os.path.join(PlotFolder, im)))
writer.close()
