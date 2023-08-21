import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import glob, os

from mrexo.mle_utils_nd import calculate_conditional_distribution, NumericalIntegrate2D

matplotlib.rcParams['xtick.labelsize'] = 25
matplotlib.rcParams['ytick.labelsize'] = 25
cmap = matplotlib.cm.viridis
cmap = matplotlib.cm.Spectral


################ Run Conditional Distribution ################ 

# Specify condition string based on the characters defined in DataDict for different dimensions. 
# This particular example is for a 3D fit.
ConditionString = 'insol,r|stm'

RunName = r"AllPlanet_RpLt4_StMlt1.5_RSStM_CV_0MC_0BS"

save_path = os.path.join(r"C:\Users\skanodia\Documents\GitHub\mrexo\sample_scripts", 'TestRuns', RunName)

ConditionName = '3D_'+ConditionString.replace('|', '_').replace(',', '_')
PlotFolder = os.path.join(save_path, ConditionName)
if not os.path.exists(PlotFolder): os.makedirs(PlotFolder)

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


xseq = DataDict['DataSequence'][LHSDimensions[0]]
yseq = DataDict['DataSequence'][LHSDimensions[1]]
zseq = DataDict['DataSequence'][RHSDimensions[0]]

r = [0.2, 0.4, 0.43, 0.46, 0.49, 0.52, 0.55, 0.57, 0.6]
MeasurementDict = {'stm':[np.log10(r),  [[np.nan, np.nan]]*len(r)]}

ConditionalDist, MeanPDF, VariancePDF = calculate_conditional_distribution(ConditionString, DataDict, weights, deg_per_dim,
	JointDist, MeasurementDict)


# Make a separate 2D plot for each RHS measurement (in this case stellar mass)
for k in np.arange(0, len(r), dtype=int):
	
	# Run to check that the 2D Joint Dist integrates to 1
	# _ = NumericalIntegrate2D(xseq, yseq, ConditionalDist[k], [xseq.min(), xseq.max()], [yseq.min(), yseq.max()])
	# print(_)
	fig = plt.figure(figsize=(8.5,6.5))
	im = plt.imshow(ConditionalDist[k], extent=(xseq.min(), xseq.max(), yseq.min(), yseq.max()), aspect='auto', origin='lower', interpolation='bicubic')
	# plt.plot(np.log10(Mass), np.log10(Radius),  'k.')
	plt.title(DataDict['ndim_label'][2]+" = {:.3f}".format(10**MeasurementDict[RHSTerms[0]][0][k]))
	plt.xlabel(DataDict['ndim_label'][LHSDimensions[0]])
	plt.ylabel(DataDict['ndim_label'][LHSDimensions[1]])

	plt.xlim(DataDict['ndim_bounds'][LHSDimensions[0]][0], DataDict['ndim_bounds'][LHSDimensions[0]][1])
	plt.ylim(DataDict['ndim_bounds'][LHSDimensions[1]][0], DataDict['ndim_bounds'][LHSDimensions[1]][1])
	plt.tight_layout()

	XTicks = np.linspace(xseq.min(), xseq.max(), 5)
	YTicks = np.linspace(yseq.min(), yseq.max(), 5)
	XLabels = np.round(10**XTicks, 1)
	YLabels = np.round(10**YTicks, 2)

	plt.xticks(XTicks, XLabels)
	plt.yticks(YTicks, YLabels)
	cbar = fig.colorbar(im, ticks=[np.min(ConditionalDist[i]), np.max(ConditionalDist[i])], fraction=0.037, pad=0.04)
	cbar.ax.set_yticklabels(['Min', 'Max'])
	plt.tight_layout()
	plt.show(block=False)


	# plt.savefig(os.path.join(PlotFolder, ConditionName+'_z_{}.png'.format(np.round(MeasurementDict[RHSTerms[0]][0][i],3))))
	plt.savefig(os.path.join(PlotFolder, ConditionName+'_z_{}.png'.format(k)))

	plt.close("all")
	
	
###############
"""
# Make a movie with the joint dists
import imageio
ListofPlots = glob.glob(os.path.join(PlotFolder, '3*.png'))
ListofPlots.sort(key=os.path.getmtime)

writer = imageio.get_writer(os.path.join(PlotFolder, ConditionName+'.gif'), duration=500)

for im in ListofPlots:
		#print(order, im)
		writer.append_data(imageio.imread(os.path.join(PlotFolder, im)))
writer.close()
# """
