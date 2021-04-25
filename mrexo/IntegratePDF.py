from scipy.integrate import simps
from scipy.interpolate import interpn
from mrexo.mle_utils_nd import calculate_conditional_distribution
from scipy.interpolate import RectBivariateSpline


def NumericalIntegrate2D(xarray, yarray, Matrix, xlimits, ylimits):
	"""
	
	"""
	
	Integral = RectBivariateSpline(xarray, yarray, Matrix).integral(
		xa=xlimits[0], xb=xlimits[1], ya=ylimits[0], yb=ylimits[1])
	# Integral2 = simps(simps(Matrix, xarray), yarray)
	return Integral

# _ = NumericalIntegrate2D(x, y, JointDist, [xseq.min(), xseq.max()], [yseq.min(), yseq.max()])
# _ = NumericalIntegrate2D(x, y, JointDist, [1.7, xseq.max()], [0.8, yseq.max()])
# _ = NumericalIntegrate2D(x, y, ConditionalDist[0], [xseq.min(), xseq.max()], [yseq.min(), yseq.max()])


# 20210310 - Checked that the 2D and 3D joint distribution does integrate to 1 using 
# RectBiVariateSpline and simpsons integrator



ConditionString = 'r,p|feh'

Condition = ConditionString.split('|')
LHSTerms = Condition[0].split(',')
RHSTerms = Condition[1].split(',')
deg_vec_per_dim = [np.arange(1, deg+1) for deg in deg_per_dim] 


LHSDimensions = np.array([(np.arange(DataDict['ndim'])[np.isin(DataDict['ndim_char'] , l)])[0] for l in LHSTerms])
RHSDimensions = np.array([(np.arange(DataDict['ndim'])[np.isin(DataDict['ndim_char'] , r)])[0] for r in RHSTerms])


xseq = outputs['DataSequence'][LHSDimensions[0]]
yseq = outputs['DataSequence'][LHSDimensions[1]]
zseq = outputs['DataSequence'][RHSDimensions[0]]



PDFIntegralJup = np.zeros(50)
PDFIntegralNep = np.zeros(50)
PDFIntegralRocky = np.zeros(50)

for i in range(50):

	print(10**(zseq[i]))
	MeasurementDict = {RHSTerms[0]:[[zseq[i]], [np.nan]]}#, 'p':[[np.log10(30)],[np.nan]]}


	ConditionalDist, MeanPDF, VariancePDF = calculate_conditional_distribution(
		ConditionString, DataDict, weights, deg_per_dim,
		JointDist, MeasurementDict)
		
	"""
	fig = plt.figure(figsize=(8.5,6.5))
	im = plt.imshow(ConditionalDist[0], 
		extent=(DataDict['ndim_bounds'][0][0], DataDict['ndim_bounds'][0][1], DataDict['ndim_bounds'][1][0], DataDict['ndim_bounds'][1][1]), 
		aspect='auto', origin='lower'); 
	# plt.errorbar(x=np.log10(DataDict['ndim_data'][0]), y=np.log10(DataDict['ndim_data'][1]), xerr=0.434*DataDict['ndim_sigma'][0]/DataDict['ndim_data'][0], yerr=0.434*DataDict['ndim_sigma'][1]/DataDict['ndim_data'][1], fmt='.', color='k', alpha=0.4)
	plt.title("Stellar Mass = {} M_sun".format(str(np.round(10**(z[i]),2))))
	# plt.ylabel("Log10 "+DataDict['ndim_label'][1]);
	# plt.xlabel("Log10 "+DataDict['ndim_label'][0]);
	plt.xlabel("Planetary Mass ($M_{\oplus}$)")
	plt.ylabel("Planetary Radius ($R_{\oplus}$)")
	plt.xlim(DataDict['ndim_bounds'][0][0], DataDict['ndim_bounds'][0][1])
	plt.ylim(DataDict['ndim_bounds'][1][0], DataDict['ndim_bounds'][1][1])
	plt.tight_layout()
	# XTicks = np.linspace(xseq.min(), xseq.max(), 5)
	XTicks = np.log10(np.array([0.3, 1, 3, 10, 30, 100, 300]))
	# YTicks = np.linspace(yseq.min(), yseq.max(), 5)
	YTicks = np.log10(np.array([1, 3, 5, 10]))


	XLabels = np.round(10**XTicks, 1)

	YLabels = np.round(10**YTicks, 2)

	plt.xticks(XTicks, XLabels)
	plt.yticks(YTicks, YLabels)
	cbar = fig.colorbar(im, ticks=[np.min(JointDist), np.max(JointDist)], fraction=0.037, pad=0.04)
	cbar.axseq.set_yticklabels(['Min', 'Max'])
	plt.tight_layout()
	plt.show(block=False)
	"""

	###############################################

	# PDFIntegralJup[i] = NumericalIntegrate2D(x, y, ConditionalDist[0], [xseq.min(), xseq.max()], [0.778, yseq.max()])
	# PDFIntegralNep[i] = NumericalIntegrate2D(x, y, ConditionalDist[0], [xseq.min(), xseq.max()], [0.3010, 0.778])
	# PDFIntegralRocky[i] = NumericalIntegrate2D(x, y, ConditionalDist[0], [xseq.min(), xseq.max()], [yseq.min(), 0.3010])

	print(NumericalIntegrate2D(xseq, yseq, ConditionalDist[0],[xseq.min(), xseq.max()], [yseq.min(), yseq.max()]))
	# Giving transpose of ConditionalDist[0] due to Python Indexing order shenanigans # 20200321
	PDFIntegralJup[i] = NumericalIntegrate2D(xseq, yseq, ConditionalDist[0].T,[0.60, xseq.max()], [yseq.min(), yseq.max()])
	PDFIntegralNep[i] = NumericalIntegrate2D(xseq, yseq, ConditionalDist[0].T, [0.3010, 0.60], [yseq.min(), yseq.max()])
	PDFIntegralRocky[i] = NumericalIntegrate2D(xseq, yseq, ConditionalDist[0].T, [xseq.min(), 0.3010], [yseq.min(), yseq.max()])


if RHSTerms[0] != 'feh':
	plt.plot(10**zseq, PDFIntegralJup, label='Gas Giant')
	plt.plot(10**zseq, PDFIntegralNep, label='Neptune')
	plt.plot(10**zseq, PDFIntegralRocky, label='Rocky')
else:
	plt.plot(zseq, PDFIntegralJup, label='Gas Giant')
	plt.plot(zseq, PDFIntegralNep, label='Neptune')
	plt.plot(zseq, PDFIntegralRocky, label='Rocky')
	
plt.xlabel(DataDict['ndim_label'][RHSDimensions[0]])
plt.ylabel("Integrated PDF ")
# plt.ylabel("Integrated PDF M>50M_E, R>6 R_E")
# plt.xlim(0.2, 0.6)
plt.title("St Teff < {} K".format(UTeff))
plt.tight_layout()
plt.legend(prop={"size":15})
plt.show(block=False)


#################################
" Run Test case to figure out transposes nonsense "

"""
SampleZ = np.zeros((40, 10))
Y = np.arange(40)
X = np.arange(10)
SampleZ[30:39, 6:10] = 1

# For the integration, give y, x and the transpose of Z 
print(NumericalIntegrate2D(X, Y, SampleZ.T, [5,10], [30,40]))

# Whereas for the plotting we can plot Z, as is, and give extent as X, Y
plt.imshow(SampleZ, origin='lower', extent=(xseq.min(), xseq.max(), yseq.min(), yseq.max()))
plt.show(block=False)
"""
