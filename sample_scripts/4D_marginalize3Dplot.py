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

ConditionString = 'm|insol,r,stm'
ConditionName = '4D_'+ConditionString.replace('|', '_').replace(',', '_')

# Result Directory for the small planet example shown in Kanodia et al. 2023 is included here - https://zenodo.org/record/8222163
RunName = r"AllPlanet_RpLt4_StMlt1.5_MRSStM_d40_0MC_0BS"

save_path = os.path.join(r"C:\Users\skanodia\Documents\GitHub\mrexo\sample_scripts", 'TestRuns', RunName)

PlotFolder = os.path.join(save_path, ConditionName)

if not os.path.exists(PlotFolder):
	print("4D Plot folder does not exist")
	os.mkdir(PlotFolder)

deg_per_dim = np.loadtxt(os.path.join(save_path, 'output', 'deg_per_dim.txt'))
DataDict = np.load(os.path.join(save_path, 'input', 'DataDict.npy'), allow_pickle=True).item()
JointDist = np.load(os.path.join(save_path, 'output', 'JointDist.npy'), allow_pickle=True).T
weights = np.loadtxt(os.path.join(save_path, 'output', 'weights.txt'))
deg_per_dim = np.loadtxt(os.path.join(save_path, 'output', 'deg_per_dim.txt')).astype(int)

Condition = ConditionString.split('|')
LHSTerms = Condition[0].split(',')
RHSTerms = Condition[1].split(',')
deg_vec_per_dim = [np.arange(1, deg+1) for deg in deg_per_dim] 

LHSDimensions = np.array([(np.arange(DataDict['ndim'])[np.isin(DataDict['ndim_char'] , l)])[0] for l in LHSTerms])
RHSDimensions = np.array([(np.arange(DataDict['ndim'])[np.isin(DataDict['ndim_char'] , r)])[0] for r in RHSTerms])

x = DataDict['DataSequence'][RHSDimensions[0]]
y = DataDict['DataSequence'][RHSDimensions[1]]
z = DataDict['DataSequence'][RHSDimensions[2]]

xdata = DataDict['ndim_data'][RHSDimensions[0]]
ydata = DataDict['ndim_data'][RHSDimensions[1]]
zdata = DataDict['ndim_data'][RHSDimensions[2]]

XTicks = np.linspace(x.min(), x.max(), 5)
XLabels = np.round(10**XTicks, 2)
XTicks = np.log10(XLabels)

YTicks = np.linspace(y.min(), y.max(), 5)
YLabels = np.round(10**YTicks, 2)
YTicks = np.log10(YLabels)

# Change to corresponding axes as required

# """
# Custom
RadiusArray = np.arange(1.5, 3.1, 0.2)
RadiusArray = [1.5, 3.0]
x = np.log10(RadiusArray)
InsolationArray = [100, 300]
InsolationArray = np.logspace(-0.1, 3, 10)
y = np.log10(InsolationArray)
StellarMassArray = np.arange(0.2, 1.2, 0.1)
StellarMassArray = [1]
z = np.log10(StellarMassArray)
# """

CombinedQuery = np.rollaxis(np.array(np.meshgrid(x, y, z)), 0, 3).reshape(len(x)*len(y)*len(z),3)


MeasurementDict=  {
			'r':[CombinedQuery[:,0], [[np.nan, np.nan]]*len(CombinedQuery)], 
			'insol':[CombinedQuery[:,1], [[np.nan, np.nan]]*len(CombinedQuery)],
			'stm':[CombinedQuery[:,2], [[np.nan, np.nan]]*len(CombinedQuery)]
}

ConditionalDist, MeanPDF, VariancePDF = calculate_conditional_distribution(
	ConditionString, DataDict, weights, deg_per_dim,
	JointDist.T, MeasurementDict)

MeanPDF = MeanPDF.reshape((len(y), len(x), len(z)))

for k in np.arange(0, len(z),  dtype=int):
	
	ChosenZ = z[k]
	print(ChosenZ)

	fig, ax1 = plt.subplots()
	im = ax1.imshow(10**MeanPDF[:,:,k], 
		extent=(x.min(), x.max(), y.min(), y.max()), 
		aspect='auto', origin='lower', interpolation='bicubic'); 
	plt.show(block=False)

	# ax1.set_title("{} = {} d".format( DataDict['ndim_label'][RHSDimensions[2]], str(np.round(10**ChosenZ,3))))
	ax1.set_ylabel(DataDict['ndim_label'][RHSDimensions[0]]);
	ax1.set_xlabel(DataDict['ndim_label'][RHSDimensions[1]]);
	ax1.set_title('Mp|Rp, Insol, StM ={:.3f} M_sun'.format(10**ChosenZ))

	XTicks = np.linspace(x.min(), x.max(), 5)
	YTicks = np.linspace(y.min(), y.max(), 5)
	XLabels = np.round(10**XTicks, 1)
	YLabels = np.round(10**YTicks, 2)

	plt.xticks(XTicks, XLabels)
	plt.yticks(YTicks, YLabels)
	
	plt.colorbar(im, label=DataDict['ndim_label'][LHSDimensions[0]])	
	plt.tight_layout()
	plt.show(block=False)
	
	plt.savefig(os.path.join(PlotFolder, ConditionName+'_z_{}.png'.format(np.round(10**ChosenZ,3))))
	plt.close()

"""
ListofPlots = glob.glob(os.path.join(PlotFolder, '4*.png'))
ListofPlots.sort(key=os.path.getmtime)

writer = imageio.get_writer(os.path.join(PlotFolder, ConditionName+'.gif'), duration=500)

for im in ListofPlots:
		#print(order, im)
		writer.append_data(imageio.imread(os.path.join(PlotFolder, im)))
writer.close()
"""

