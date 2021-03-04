import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import imageio
import glob
from mrexo.mle_utils_nd import calculate_conditional_distribution

ConditionString = 'r|stm,feh,p'
ConditionName = '4D_'+ConditionString.replace('|', '_').replace(',', '_')

PlotFolder = os.path.join(save_path, ConditionName)

if not os.path.exists(PlotFolder):
	print("4D Plot folder does not exist")
	os.mkdir(PlotFolder)



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
# XLabels = np.array([0.7, 1, 3, 5, 10])
XTicks = np.log10(XLabels)

YTicks = np.linspace(y.min(), y.max(), 5)
YLabels = np.round(10**YTicks, 2)
YLabels = np.round(YTicks, 2) # For Metallicity 

for k in np.arange(0, len(z), 2, dtype=int):
	
	ChosenZ = z[k]
	print(ChosenZ)
	MeanPDF = np.zeros((len(x), len(y)))
	VariancePDF = np.zeros((len(x), len(y)))

	for i in range(len(x)):
		for j in range(len(y)):
			MeasurementDict = {RHSTerms[0]:[[x[i]], [np.nan]], RHSTerms[1]:[[y[j]], [np.nan]], RHSTerms[2]:[[ChosenZ],[np.nan]]}

			ConditionalDist, MeanPDF[i,j], VariancePDF[i,j] = calculate_conditional_distribution(
				ConditionString, DataDict, weights, deg_per_dim,
				JointDist.T, MeasurementDict)
			# print(MeanPDF)

	###########################
	
	fig, ax1 = plt.subplots()

	im = ax1.imshow(10**MeanPDF, 
		extent=(x.min(), x.max(), y.min(), y.max()), 
		aspect='auto', origin='lower', interpolation='bicubic'); 

	# plt.plot(np.log10(DataDict['ndim_data'][0]), np.log10(DataDict['ndim_data'][1]), 'k.')
	# ax1.set_title("{} = {} d".format( DataDict['ndim_label'][RHSDimensions[2]], str(np.round(10**ChosenZ,3))))
	ax1.set_xlabel(DataDict['ndim_label'][RHSDimensions[0]]);
	ax1.set_ylabel(DataDict['ndim_label'][RHSDimensions[1]]);
	ax1.set_title('Rp|StM, Fe/H, P={} d'.format(int(10**ChosenZ)))
	
	plt.colorbar(im, label=DataDict['ndim_label'][LHSDimensions[0]])
	
	# ZMask = (zdata > 0.5*(10**ChosenZ)) & (zdata < 2*(10**ChosenZ))
	ZMask = zdata <= 5
	# ZMask = (zdata > 5) & (zdata <= 10)
	Zmask = zdata >10
	# ZMask = np.ones(len(zdata), dtype=bool)
	xdataMasked = xdata[ZMask]
	ydataMasked = ydata[ZMask]
	Histogram = np.histogram2d(np.log10(xdataMasked), np.log10(ydataMasked))
	HistogramMask = np.ones((np.shape(Histogram[0])))  
	HistogramMask = np.ma.masked_where(Histogram[0] > 0, HistogramMask)

	ax1.imshow(HistogramMask, 
		extent=(np.log10(xdata.min()), np.log10(xdata.max()), np.log10(ydata.min()), np.log10(ydata.max())),
		# extent=(x.min(), x.max(), y.min(), y.max()),
		 aspect='auto', origin='lower',
		alpha=0.4, cmap='binary_r')	

	if RHSTerms[0] == 'stm':
		XTicks = np.log10(np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
		XLabels = np.round(10**XTicks, 2)
		plt.xticks(XTicks, XLabels)
		ax1.set_xticks(XTicks)
		ax1.set_xticklabels(XLabels)
		
		ax1.set_xlim(np.log10(0.06), np.log10(0.6))
		
		ax2 = ax1.twiny()
		ax2.set_xlim(ax1.get_xlim())
		# formatter = FuncFormatter(lambda x, pos: '{:0.2f}'.format(np.sqrt(x)))
		# ax2.xaxis.set_major_formatter(formatter)

		ax2Ticks = np.log10(np.array([0.6, 0.39, 0.2, 0.1, 0.077]))
		ax2Labels = ['M0', 'M3', 'M5', 'M7', 'M9']
		# ax2.set_xticks(XTicks)
		# ax2.set_xticklabels(XLabels)
		ax2.set_xticks(ax2Ticks)
		ax2.set_xticklabels(ax2Labels)
		
	else:
		XTicks = np.linspace(x.min(), x.max(), 5)
		XLabels = np.round(10**XTicks, 2)
		plt.xticks(XTicks, XLabels)
		
		plt.xlim(np.log10(xdata.min()), np.log10(xdata.max()))

	if RHSTerms[1] == 'feh':
		YTicks = np.linspace(-0.5, 0.5, 5)
		YLabels = YTicks
		plt.yticks(YTicks, YLabels)
		plt.ylim(-0.55, 0.45)

	else:
		YTicks = np.linspace(y.min(), y.max(), 5)
		YLabels = np.round(10**YTicks, 2)
		plt.yticks(YTicks, YLabels)
		plt.ylim(np.log10(ydata.min()), np.log10(ydata.max()))
	
	plt.tight_layout()
	plt.show(block=False)
	
	plt.savefig(os.path.join(PlotFolder, ConditionName+'_z_{}.png'.format(np.round(ChosenZ,3))))
	plt.close()

"""
ListofPlots = glob.glob(os.path.join(PlotFolder, '4*.png'))
ListofPlots.sort(key=os.path.getmtime)

writer = imageio.get_writer(os.path.join(PlotFolder, ConditionName+'.mp4'), fps=2)

for im in ListofPlots:
		#print(order, im)
		writer.append_data(imageio.imread(os.path.join(PlotFolder, im)))
writer.close()
"""

