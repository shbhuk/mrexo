import matplotlib.pyplot as plt
import numpy as np

from mrexo.mle_utils_nd import calculate_conditional_distribution

ConditionString = 'p|r,m'
ConditionString = 'm|stm,r,p'

Condition = ConditionString.split('|')
LHSTerms = Condition[0].split(',')
RHSTerms = Condition[1].split(',')
deg_vec_per_dim = [np.arange(1, deg+1) for deg in deg_per_dim] 


LHSDimensions = np.arange(DataDict['ndim'])[np.isin(DataDict['ndim_char'] , LHSTerms)]
RHSDimensions = np.arange(DataDict['ndim'])[np.isin(DataDict['ndim_char'] , RHSTerms)]
 
x = DataDict['DataSequence'][RHSDimensions[0]]
y = DataDict['DataSequence'][RHSDimensions[1]]

MeanPDF = np.zeros((len(x), len(y)))
VariancePDF = np.zeros((len(x), len(y)))



for i in range(len(x)):
	for j in range(len(y)):
		MeasurementDict = {RHSTerms[0]:[[x[i]], [np.nan]], RHSTerms[1]:[[y[j]], [np.nan]], 'p':[[np.log10(10)],[np.nan]]}

		ConditionalDist, MeanPDF[i,j], VariancePDF[i,j] = calculate_conditional_distribution(
			ConditionString, DataDict, weights, deg_per_dim,
			JointDist.T, MeasurementDict)
		# print(MeanPDF)


XTicks = np.linspace(x.min(), x.max(), 5)
XLabels = np.round(10**XTicks, 2)
XLabels = np.array([0.7, 1, 3, 5, 10, 15])
XTicks = np.log10(XLabels)

YTicks = np.linspace(y.min(), y.max(), 5)
YLabels = np.round(10**YTicks, 2)

plt.figure()
plt.imshow(10**MeanPDF, 
	extent=(x.min(), x.max(), y.min(), y.max()), 
	aspect='auto', origin='lower', interpolation='bicubic'); 
# plt.plot(np.log10(DataDict['ndim_data'][0]), np.log10(DataDict['ndim_data'][1]), 'k.')
# plt.title("Orbital Period = {} d".format(str(np.round(title,3))))
plt.xlabel(DataDict['ndim_label'][RHSDimensions[0]]);
plt.ylabel(DataDict['ndim_label'][RHSDimensions[1]]);

plt.colorbar(label=DataDict['ndim_label'][LHSDimensions[0]])
plt.tight_layout()
plt.title('Period = 10 days')
plt.xticks(XTicks, XLabels)
plt.yticks(YTicks, YLabels)
plt.tight_layout()
plt.show(block=False)
