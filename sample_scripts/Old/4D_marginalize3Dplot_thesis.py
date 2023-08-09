import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
# import imageio
import glob, os
from mrexo.mle_utils_nd import calculate_conditional_distribution

# ConditionString = 'r|stm,feh,p'
ConditionString = 'm|r,stm,insol'
ConditionName = '4D_'+ConditionString.replace('|', '_').replace(',', '_')


RunName = "Mdwarf_4D_deg45_20220409_M_R_S_StM1.2_bounded"
# RunName = "Fake_4D_MRSStM"
save_path = os.path.join(r"C:\Users\shbhu\Documents\GitHub\mrexo\sample_scripts", 'TestRuns', 'ThesisRuns', RunName)

PlotFolder = os.path.join(save_path, ConditionName)

if not os.path.exists(PlotFolder):
	print("4D Plot folder does not exist")
	os.mkdir(PlotFolder)

deg_per_dim = np.loadtxt(os.path.join(save_path, 'output', 'deg_per_dim.txt'))
DataDict = np.load(os.path.join(save_path, 'input', 'DataDict.npy'), allow_pickle=True).item()
JointDist = np.load(os.path.join(save_path, 'output', 'JointDist.npy'), allow_pickle=True)
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
# XLabels = np.array([0.7, 1, 3, 5, 10])
XLabels = np.array([0.3, 1, 10, 50, 100])
XTicks = np.log10(XLabels)

YTicks = np.linspace(y.min(), y.max(), 5)
YLabels = np.round(10**YTicks, 2)
YLabels = np.array([0.7, 1, 3, 5, 10])
YLabels = np.array([1, 2, 3, 4])
YTicks = np.log10(YLabels)
# YLabels = np.round(YTicks, 2) # For Metallicity 


MeasurementDict = {'stm':[[0.6, 0.75, 1.0], [np.nan]*4], 'r':[[12], [np.nan]], 'insol':[[10, 30, 60, 100, 300], [np.nan]]}
# MeasurementDict = {'stm':[[0.3], [np.nan]*1], 'r':[[50], [np.nan]], 'insol':[[1], [np.nan]]}

cmap =matplotlib.cm.Spectral
# Establish colour range based on variable
norm = matplotlib.colors.Normalize(vmin=0.5, vmax=1.15)

fig, ax = plt.subplots(5, sharex=True, sharey=False, figsize=(7,9), dpi=200)


for j in range(len(MeasurementDict[RHSTerms[2]][0])): # Insolation
	
	# fig, ax = plt.subplots()#1, sharex=True, sharey=True, figsize=(1,6.5))
	for k in range(len(MeasurementDict[RHSTerms[1]][0])): # Stellar Mass
		
		colour = cmap(norm(MeasurementDict[RHSTerms[1]][0][k]))
		LogMeasurementDict = {RHSTerms[0]:[[np.log10(MeasurementDict[RHSTerms[0]][0][0])], [np.nan]], 
			RHSTerms[1]:[[np.log10(MeasurementDict[RHSTerms[1]][0][k])], [np.nan]], 
			RHSTerms[2]:[[np.log10(MeasurementDict[RHSTerms[2]][0][j])], [np.nan]]}
	
		ConditionalDist, MeanPDF, VariancePDF = calculate_conditional_distribution(ConditionString, DataDict, weights, deg_per_dim,
			JointDist, LogMeasurementDict)
			
		ax[j].plot(10**xseq, ConditionalDist[0]/np.sum(ConditionalDist[0]), label=r'St. Mass = {:.2f} '.format(MeasurementDict[RHSTerms[1]][0][k])+' M$_{\odot}$', c=colour)
		ax[j].axvline(10**MeanPDF[0], linestyle='dashed', c=colour)
		# ax[j].text(
		ax[j].text(1000, 0.05, str(MeasurementDict[RHSTerms[2]][0][j]) + '$ S_{\oplus}$', fontsize=18)

		
			
		
	XTicks = [10, 30, 100, 300, 1000]
	# plt.ylim(0, 20)
	ax[j].grid(alpha=0.3)
	ax[j].set_xticks(XTicks)
	ax[j].set_xscale("log")
	ax[j].set_xticklabels(["{:1d}".format(s) for s in XTicks])
	ax[j].set(yticklabels=[])  # remove the tick labels
	# ax[j].set_xlim(100, 2000)
	# fig.hspace(0)

plt.xlabel("Planet Mass ($M_{\oplus}$)")
ax[2].set_ylabel("Probability Density Function")
ax[0].set_title("Expected Planetary Mass \nRp =12 Re vs Host Star Mass vs Insolation", fontsize=15)
ax[0].legend()

# plt.tight_layout()
fig.subplots_adjust(hspace=0.01)
plt.show(block=False)
	

"""
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
	# ax1.set_title('Rp|StM, Fe/H, Insol={} S_earth'.format(int(10**ChosenZ)))
	ax1.set_title('Mp|Rp, Insol, StM ={:.3f} M_sun'.format(10**ChosenZ))

	
	plt.colorbar(im, label=DataDict['ndim_label'][LHSDimensions[0]])
	
	# ZMask = (zdata > 0.5*(10**ChosenZ)) & (zdata < 2*(10**ChosenZ))
	ZMask = zdata <= 1
	# ZMask = (zdata > 5) & (zdata <= 10)
	# Zmask = zdata >10
	ZMask = np.ones(len(zdata), dtype=bool)
	xdataMasked = xdata[ZMask]
	ydataMasked = ydata[ZMask]
	Histogram = np.histogram2d(np.log10(xdataMasked), np.log10(ydataMasked))
	HistogramMask = np.ones((np.shape(Histogram[0])))  
	HistogramMask = np.ma.masked_where(Histogram[0] > 0, HistogramMask)

	'''	
	ax1.imshow(HistogramMask, 
		extent=(np.log10(xdata.min()), np.log10(xdata.max()), np.log10(ydata.min()), np.log10(ydata.max())),
		# extent=(x.min(), x.max(), y.min(), y.max()),
		 aspect='auto', origin='lower',
		alpha=0.4, cmap='binary_r')	
	'''

	if RHSTerms[0] == 'stm':
		XTicks = np.log10(np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
		XLabels = np.round(10**XTicks, 2)
		plt.xticks(XTicks, XLabels)
		ax1.set_xticks(XTicks)
		ax1.set_xticklabels(XLabels)
		
		ax1.set_xlim(np.log10(0.08), np.log10(0.6))
		
		ax2 = ax1.twiny()
		ax2.set_xlim(ax1.get_xlim())
		# formatter = FuncFormatter(lambda x, pos: '{:0.2f}'.format(np.sqrt(x)))
		# ax2.xaxis.set_major_formatter(formatter)

		ax2Ticks = np.log10(np.array([0.6, 0.39, 0.2, 0.1]))
		ax2Labels = ['M0', 'M3', 'M5', 'M7']
		# ax2.set_xticks(XTicks)
		# ax2.set_xticklabels(XLabels)
		ax2.set_xticks(ax2Ticks)
		ax2.set_xticklabels(ax2Labels)
		
	else:
		# XTicks = np.linspace(x.min(), x.max(), 5)
		# XLabels = np.round(10**XTicks, 2)
		plt.xticks(XTicks, XLabels)
		# plt.xlim(x.min(), x.max())
		plt.xlim(np.log10(0.3), np.log10(100))

	if RHSTerms[1] == 'feh':
		YTicks = np.linspace(-0.5, 0.5, 5)
		YLabels = YTicks
		plt.yticks(YTicks, YLabels)
		plt.ylim(-0.55, 0.45)

	else:
		# YTicks = np.linspace(y.min(), y.max(), 5)
		# YLabels = np.round(10**YTicks, 2)
		plt.yticks(YTicks, YLabels)
		plt.ylim(np.log10(ydata.min()), np.log10(ydata.max()))
	
	plt.tight_layout()
	plt.show(block=False)
	
	plt.savefig(os.path.join(PlotFolder, ConditionName+'_z_{}.png'.format(np.round(10**ChosenZ,3))))
	plt.close()
"""

"""
ListofPlots = glob.glob(os.path.join(PlotFolder, '4*.png'))
ListofPlots.sort(key=os.path.getmtime)

writer = imageio.get_writer(os.path.join(PlotFolder, ConditionName+'.mp4'), fps=2)

for im in ListofPlots:
		#print(order, im)
		writer.append_data(imageio.imread(os.path.join(PlotFolder, im)))
writer.close()
"""

