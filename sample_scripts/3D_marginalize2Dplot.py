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

ConditionString = 'm|insol,r'

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

x = DataDict['DataSequence'][RHSDimensions[0]]
y = DataDict['DataSequence'][RHSDimensions[1]]

xdata = DataDict['ndim_data'][RHSDimensions[0]]
ydata = DataDict['ndim_data'][RHSDimensions[1]]

# Change to corresponding axes as required
RadiusArray = np.arange(1.5, 3.1, 0.2)
InsolationArray = [100, 300]
InsolationArray = np.logspace(-0.1, 3)

CombinedQuery = np.rollaxis(np.array(np.meshgrid(RadiusArray, InsolationArray)), 0, 3).reshape(len(RadiusArray)*len(InsolationArray),2)
MeasurementDict=  {
			'r':[np.log10(CombinedQuery[:,0]), [[np.nan, np.nan]]*len(CombinedQuery)], 
			'insol':[np.log10(CombinedQuery[:,1]), [[np.nan, np.nan]]*len(CombinedQuery)]
}

ConditionalDist, MeanPDF, VariancePDF = calculate_conditional_distribution(
	ConditionString, DataDict, weights, deg_per_dim,
	JointDist.T, MeasurementDict)

# XTicks = np.linspace(x.min(), x.max(), 5)
# XLabels = np.round(10**XTicks, 2)
# XTicks =np.log10([1, 3, 10, 30, 100])
# XLabels = np.array([0.7, 1, 3, 5, 10]))


plt.figure()
plt.imshow(10**np.reshape(MeanPDF, (len(InsolationArray), len(RadiusArray))),
	extent=(x.min(), x.max(), y.min(), y.max()), 
	aspect='auto', origin='lower', interpolation='bicubic'); 
plt.xlabel(DataDict['ndim_label'][RHSDimensions[0]]);
plt.ylabel(DataDict['ndim_label'][RHSDimensions[1]]);

plt.colorbar(label=DataDict['ndim_label'][LHSDimensions[0]])
plt.xticks(XTicks, XLabels)
plt.yticks(YTicks, YLabels)

plt.title(ConditionString)

plt.tight_layout()
plt.show(block=False)
