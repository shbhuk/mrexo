import matplotlib.pyplot as plt
import os, sys
import numpy as np
import matplotlib


matplotlib.rcParams['xtick.labelsize'] = 25
matplotlib.rcParams['ytick.labelsize'] = 25

save_path = r"C:\Users\skanodia\Documents\GitHub\mrexo\examples\CKS-X\CKS-X_radius_stm"

# _ = Plot2DJointDistribution(save_path)
# _ = Plot2DWeights(save_path)

# plt.close("all")


def Plot1DInputDataHistogram(save_path):
	"""
	Plot a 1D histogram for the data in each dimension and save the figures in ``save_path``.
	"""

	DataDict = np.load(os.path.join(save_path, 'input', 'DataDict.npy'), allow_pickle=True).item()
	ndim = DataDict['ndim']
	for n in range(ndim):
		x = DataDict['ndim_data'][n]
		plt.hist(x, bins=np.logspace(np.log10(x.min()), np.log10(x.max()), 20+1))
		plt.xlabel(DataDict['ndim_label'][n])
		plt.ylabel("Number #")
		plt.gca().set_xscale("log")
		# plt.title(RunName)
		plt.tight_layout()
		plt.savefig(os.path.join(save_path, 'output', 'Histogram_'+DataDict['ndim_char'][n]+'.png'))
		plt.close("all")

def Plot2DWeights(save_path):
	"""
	Plot a 2D heat-map of the weights and save the figure in ``save_path``.
	"""

	deg_per_dim = np.loadtxt(os.path.join(save_path, 'output', 'deg_per_dim.txt')).astype(int)
	weights = np.loadtxt(os.path.join(save_path, 'output', 'weights.txt'))

	fig = plt.figure(figsize=(8.5,6.5))
	plt.imshow(np.reshape(weights , deg_per_dim).T, origin = 'lower', aspect='auto')
	plt.title(deg_per_dim)
	plt.colorbar()
	plt.savefig(os.path.join(save_path, 'output', 'Weights.png'))

	return fig

def Plot2DJointDistribution(save_path):
	"""
	Plot a 2D heat-map of the joint distribution with the input data overlaid, and save the figure in ``save_path``.
	"""

	deg_per_dim = np.loadtxt(os.path.join(save_path, 'output', 'deg_per_dim.txt')).astype(int)
	DataDict = np.load(os.path.join(save_path, 'input', 'DataDict.npy'), allow_pickle=True).item()
	DataSequences = np.loadtxt(os.path.join(save_path, 'output', 'other_data_products', 'DataSequences.txt'))
	weights = np.loadtxt(os.path.join(save_path, 'output', 'weights.txt'))
	JointDist = np.load(os.path.join(save_path, 'output', 'JointDist.npy'), allow_pickle=True)
	cmap = matplotlib.cm.YlGnBu


	################ Plot Joint Distribution ################ 
	x = DataDict['DataSequence'][0]
	y = DataDict['DataSequence'][1]

	fig = plt.figure(figsize=(8.5,6.5))
	im = plt.imshow(JointDist.T, 
		extent=(DataDict['ndim_bounds'][0][0], DataDict['ndim_bounds'][0][1], DataDict['ndim_bounds'][1][0], DataDict['ndim_bounds'][1][1]), 
		aspect='auto', origin='lower', cmap=cmap); 
	plt.errorbar(x=np.log10(DataDict['ndim_data'][0]), y=np.log10(DataDict['ndim_data'][1]), xerr=(0.434*DataDict['ndim_LSigma'][0]/DataDict['ndim_data'][0], 0.434*DataDict['ndim_USigma'][0]/DataDict['ndim_data'][0]), 
							yerr=(0.434*DataDict['ndim_LSigma'][1]/DataDict['ndim_data'][1], 0.434*DataDict['ndim_USigma'][1]/DataDict['ndim_data'][1]),
							fmt='.', color='k', alpha=0.4)
	plt.ylabel(DataDict['ndim_label'][1]);
	plt.xlabel(DataDict['ndim_label'][0]);
	plt.xlim(DataDict['ndim_bounds'][0][0], DataDict['ndim_bounds'][0][1])
	plt.ylim(DataDict['ndim_bounds'][1][0], DataDict['ndim_bounds'][1][1])
	plt.tight_layout()

	XTicks = np.linspace(x.min(), x.max(), 5)
	YTicks = np.linspace(y.min(), y.max(), 5)
	XLabels = np.round(10**XTicks, 1)
	YLabels = np.round(10**YTicks, 2)

	plt.xticks(XTicks, XLabels)
	plt.yticks(YTicks, YLabels)
	cbar = fig.colorbar(im, ticks=[np.min(JointDist), np.max(JointDist)], fraction=0.037, pad=0.04)
	cbar.ax.set_yticklabels(['Min', 'Max'])
	plt.tight_layout()
	plt.savefig(os.path.join(save_path, 'output', 'JointDist.png'), dpi=360)

	return fig
