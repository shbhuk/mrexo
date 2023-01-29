import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np

from mrexo.mle_utils_nd import calculate_conditional_distribution

ConditionString = 'm|r,p'
ConditionString = 'r|stm,feh'
ConditionString = 'r|p,stm'
ConditionString = 'm|r,insol'

RunName = r"Trial_FGKM_3D_MRS"


save_path = os.path.join(r"C:\Users\skanodia\Documents\GitHub\mrexo\sample_scripts", 'TestRuns', RunName)

ConditionName = '3D_'+ConditionString.replace('|', '_').replace(',', '_')
PlotFolder = os.path.join(save_path, ConditionName)

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

MeanPDF = np.zeros((len(x), len(y)))
VariancePDF = np.zeros((len(x), len(y)))




for i in range(len(x)):
	for j in range(len(y)):
		MeasurementDict = \
		{
			RHSTerms[0]:[[x[i]], [[np.nan, np.nan]]], 
			RHSTerms[1]:[[y[j]], [[np.nan, np.nan]]]
		}#, 'p':[[np.log10(30)],[np.nan]]}


		ConditionalDist, MeanPDF[i,j], VariancePDF[i,j] = calculate_conditional_distribution(
			ConditionString, DataDict, weights, deg_per_dim,
			JointDist.T, MeasurementDict)
		# print(MeanPDF)


XTicks = np.linspace(x.min(), x.max(), 5)
# XTicks =np.log10([1, 3, 10, 30, 100])
XLabels = np.round(10**XTicks, 2)
# XLabels = np.array([0.7, 1, 3, 5, 10])
XTicks = np.log10(XLabels)

YTicks = np.linspace(y.min(), y.max(), 5)
# YTicks = np.log10(np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
YLabels = np.round(10**YTicks, 2)
# YLabels = np.round(YTicks, 2) # For Metallicity 


plt.figure()
plt.imshow(10**MeanPDF, 
	extent=(x.min(), x.max(), y.min(), y.max()), 
	aspect='auto', origin='lower', interpolation='bicubic'); 
# plt.plot(np.log10(DataDict['ndim_data'][0]), np.log10(DataDict['ndim_data'][1]), 'k.')
# plt.title("Orbital Period = {} d".format(str(np.round(title,3))))
plt.xlabel(DataDict['ndim_label'][RHSDimensions[0]]);
plt.ylabel(DataDict['ndim_label'][RHSDimensions[1]]);

plt.colorbar(label=DataDict['ndim_label'][LHSDimensions[0]])
# plt.tight_layout()
# plt.title('Period = 30 days')
plt.xticks(XTicks, XLabels)
plt.yticks(YTicks, YLabels)

"""
Histogram = np.histogram2d(np.log10(DataDict['ndim_data'][RHSDimensions[0]]), np.log10(DataDict['ndim_data'][RHSDimensions[1]]), bins=20)
HistogramMask = np.ones((np.shape(Histogram[0])))  
HistogramMask = np.ma.masked_where(Histogram[0] > 0, HistogramMask)

plt.imshow(HistogramMask, 
	extent=(np.log10(DataDict['ndim_data'][RHSDimensions[0]].min()), np.log10(DataDict['ndim_data'][RHSDimensions[0]].max()), 
	np.log10(DataDict['ndim_data'][RHSDimensions[1]].min()), np.log10(DataDict['ndim_data'][RHSDimensions[1]].max())),
	 aspect='auto', origin='lower',
	alpha=0.5, cmap='binary_r')
"""
plt.title(ConditionString)

plt.tight_layout()
plt.show(block=False)
