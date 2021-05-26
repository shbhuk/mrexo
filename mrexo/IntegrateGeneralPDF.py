from scipy.integrate import simps
from scipy.interpolate import interpn
from mrexo.mle_utils_nd import calculate_conditional_distribution
from scipy.interpolate import RectBivariateSpline, UnivariateSpline


def NumericalIntegrate2D(xarray, yarray, Matrix, xlimits, ylimits):
	"""
	
	"""
	
	Integral = RectBivariateSpline(xarray, yarray, Matrix).integral(
		xa=xlimits[0], xb=xlimits[1], ya=ylimits[0], yb=ylimits[1])
	# Integral2 = simps(simps(Matrix, xarray), yarray)
	return Integral
	
def NumericalIntegrate1D(xarray, Matrix, xlimits):
	Integral = UnivariateSpline(xarray, Matrix).integral(xlimits[0], xlimits[1])
	return Integral

def CalculateMarginalizeDist(ConditionString, 
	DataDict, weights, deg_per_dim, JointDist):
	
	Condition = ConditionString.split('|')
	LHSTerms = Condition[0].split(',')
	RHSTerms = Condition[1].split(',')
	
	LHSDimensions = np.array([(np.arange(DataDict['ndim'])[np.isin(DataDict['ndim_char'] , l)])[0] for l in LHSTerms])
	RHSDimensions = np.array([(np.arange(DataDict['ndim'])[np.isin(DataDict['ndim_char'] , r)])[0] for r in RHSTerms])

	print("Computing conditional distribution " + ConditionString)
	
	if len(RHSTerms) == 1:
		ConditionalDist, MeanPDF, VariancePDF = Marginalize1D(ConditionString, 
			DataDict, weights, deg_per_dim, JointDist,
			LHSDimensions, RHSDimensions,
			LHSTerms, RHSTerms)
	elif len(RHSTerms) == 2:
		ConditionalDist, MeanPDF, VariancePDF = Marginalize2D(ConditionString, 
			DataDict, weights, deg_per_dim, JointDist,
			LHSDimensions, RHSDimensions,
			LHSTerms, RHSTerms)	
			
	return ConditionalDist, MeanPDF, VariancePDF
	
def Marginalize2D(ConditionString, 
	DataDict, weights, deg_per_dim, JointDist,
	LHSDimensions, RHSDimensions,
	LHSTerms, RHSTerms):
	"""
	Marginalize over 2 dimensions 
	Conditional distribution
	x|y,z
	"""
		
	deg_vec_per_dim = [np.arange(1, deg+1) for deg in deg_per_dim] 
	xseq = outputs['DataSequence'][RHSDimensions[0]]
	yseq = outputs['DataSequence'][RHSDimensions[1]]
	
	ConditionalDist = np.zeros(( [len(xseq)]*ndim ))
	MeanPDF, VariancePDF = [np.zeros((len(xseq), len(yseq), len(LHSTerms))) for _ in range(2)]
	
	for i in range(len(xseq)):
		for j in range(len(yseq)):

			MeasurementDict = {
				RHSTerms[0]:[[xseq[i]], [np.nan]], 
				RHSTerms[1]:[[yseq[j]], [np.nan]]}

			ConditionalDist[i,j], MeanPDF[i,j], VariancePDF[i,j] = calculate_conditional_distribution(
				ConditionString, DataDict, weights, deg_per_dim,
				JointDist.T, MeasurementDict)

	return ConditionalDist, MeanPDF, VariancePDF

def Marginalize1D(ConditionString, 
	DataDict, weights, deg_per_dim, JointDist,
	LHSDimensions, RHSDimensions,
	LHSTerms, RHSTerms):
	"""
	Marginalize over 1 dimension
	Conditional distribution
	x,y|z or x|z
	"""
		
		
	deg_vec_per_dim = [np.arange(1, deg+1) for deg in deg_per_dim] 
		
	xseq = outputs['DataSequence'][RHSDimensions[0]]
	
	ConditionalDist = np.zeros(( [len(xseq)]*ndim ))
	MeanPDF, VariancePDF = [np.zeros((len(xseq), len(LHSTerms))) for _ in range(2)]
	
	for i in range(len(xseq)):
		MeasurementDict = {
			RHSTerms[0]:[[xseq[i]], [np.nan]]}

		ConditionalDist[i], MeanPDF[i], VariancePDF[i] = calculate_conditional_distribution(
			ConditionString, DataDict, weights, deg_per_dim,
			JointDist, MeasurementDict)

	return ConditionalDist, MeanPDF, VariancePDF


def Integrate1D_Marginalize1D(
	ConditionString, LHSTerms, RHSTerms, LHSDimensions, RHSDimensions,
	DataDict, 
	ConditionalDist, IntegrationRegion):
	"""
	ndim == 2
	Marginalize over 1 axis, and then integrate over 1.
	Output will be a 1 dimensional array of PDFs as a function of RHS
	x|z
	"""
	LHSSeq1 = DataDict['DataSequence'][LHSDimensions[0]]
	RHSSeq1 = DataDict['DataSequence'][RHSDimensions[0]]

	i1 = IntegrationRegion[LHSTerms[0]]

	IntegratedPDF = np.zeros(len(RHSSeq1))
	
	for r1 in range(len(RHSSeq1)):
		print(NumericalIntegrate1D(LHSSeq1, 
			ConditionalDist[r1],
			[LHSSeq1.min(), LHSSeq1.max()]))
		
		IntegratedPDF[r1] = NumericalIntegrate1D(LHSSeq1, ConditionalDist[r1].T,i1)

	return RHSSeq1, IntegratedPDF


def Integrate2D_Marginalize1D(
	ConditionString, LHSTerms, RHSTerms, LHSDimensions, RHSDimensions,
	DataDict, 
	ConditionalDist, IntegrationRegion):
	"""
	ndim == 3
	Marginalize over 1 axis, and then integrate over 2.
	Output will be a 1 dimensional array of PDFs as a function of RHS
	x, y|z
	"""
	LHSSeq1 = DataDict['DataSequence'][LHSDimensions[0]]
	LHSSeq2 = DataDict['DataSequence'][LHSDimensions[1]]
	RHSSeq1 = DataDict['DataSequence'][RHSDimensions[0]]

	i1 = IntegrationRegion[LHSTerms[0]]
	i2 = IntegrationRegion[LHSTerms[1]]

	IntegratedPDF = np.zeros(len(RHSSeq1))
	
	for r1 in range(len(RHSSeq1)):
		print(NumericalIntegrate2D(LHSSeq1, LHSSeq2, 
			ConditionalDist[r1],
			[LHSSeq1.min(), LHSSeq1.max()], [LHSSeq2.min(), LHSSeq2.max()]))
		
		IntegratedPDF[r1] = NumericalIntegrate2D(LHSSeq1, LHSSeq2, ConditionalDist[r1].T,i1, i2)

	return RHSSeq1, IntegratedPDF
	
	
def Integrate1D_Marginalize2D(
	ConditionString, LHSTerms, RHSTerms, LHSDimensions, RHSDimensions,
	DataDict, 
	ConditionalDist, IntegrationRegion):
	"""
	ndim == 3
	Marginalize over 2 axes, and then integrate over 1.
	Output will be a 2 dimensional array of PDFs as a function of RHS
	x|z
	"""
	LHSSeq1 = DataDict['DataSequence'][LHSDimensions[0]]
	RHSSeq1 = DataDict['DataSequence'][RHSDimensions[0]]
	RHSSeq2 = DataDict['DataSequence'][RHSDimensions[1]]

	i1 = IntegrationRegion[LHSTerms[0]]

	IntegratedPDF = np.zeros((len(RHSSeq1), len(RHSSeq2)))
	
	for r1 in range(len(RHSSeq1)):
		for r2 in range(len(RHSSeq2)):
			# print(NumericalIntegrate1D(LHSSeq1, 
				# ConditionalDist[r1, r2].T,
				# [LHSSeq1.min(), LHSSeq1.max()]))
		
			IntegratedPDF[r1, r2] = NumericalIntegrate1D(LHSSeq1, ConditionalDist[r1, r2].T,i1)

	return np.array([RHSSeq1, RHSSeq2]), IntegratedPDF


def Integrate2D_Marginalize2D(
	ConditionString, LHSTerms, RHSTerms, LHSDimensions, RHSDimensions,
	DataDict, 
	ConditionalDist, IntegrationRegion):
	"""
	ndim == 4
	Marginalize over 2 axes, and then integrate over 2.
	Output will be a 2 dimensional array of PDFs as a function of RHS
	x|z
	"""
	LHSSeq1 = DataDict['DataSequence'][LHSDimensions[0]]
	LHSSeq2 = DataDict['DataSequence'][LHSDimensions[0]]

	RHSSeq1 = DataDict['DataSequence'][RHSDimensions[0]]
	RHSSeq2 = DataDict['DataSequence'][RHSDimensions[1]]

	i1 = IntegrationRegion[LHSTerms[0]]
	i2 = IntegrationRegion[LHSTerms[1]]

	IntegratedPDF = np.zeros((len(RHSSeq1), len(RHSSeq2)))
	
	for r1 in range(len(RHSSeq1)):
		for r2 in range(len(RHSSeq1)):
			# print(NumericalIntegrate2D(LHSSeq1, LHSSeq2, 
				# ConditionalDist[r1, r2],
				# [LHSSeq1.min(), LHSSeq1.max()], [LHSSeq2.min(), LHSSeq2.max()]))
		
			IntegratedPDF[r1, r2] = NumericalIntegrate2D(LHSSeq1, LHSSeq2, ConditionalDist[r1, r2].T,i1, i2)

	return np.array([RHSSeq1, RHSSeq2]), IntegratedPDF


def IntegrateConditionalDistribution(
	ConditionString, IntegrationBounds,
	DataDict, weights, deg_per_dim, JointDist
	):
	"""
	First marginalizes the joint distribution over the RHS terms
	And then it integrates over the LHS terms.
	
	"""

	ConditionalDist, MeanPDF, VariancePDF = CalculateMarginalizeDist(
		ConditionString=ConditionString, 
		DataDict=DataDict, weights=weights, 
		deg_per_dim=deg_per_dim, 
		JointDist=JointDist
		)

	Condition = ConditionString.split('|')
	LHSTerms = Condition[0].split(',')
	RHSTerms = Condition[1].split(',')

	LHSDimensions = np.array([(np.arange(DataDict['ndim'])[np.isin(DataDict['ndim_char'] , l)])[0] for l in LHSTerms])
	RHSDimensions = np.array([(np.arange(DataDict['ndim'])[np.isin(DataDict['ndim_char'] , r)])[0] for r in RHSTerms])


	# Separate out the Integration bounds
	NumIntegrations = len(IntegrationBounds[list(IntegrationBounds.keys())[0]])
	IntegrationRegions = [{k:IntegrationBounds[k][i] for k in IntegrationBounds.keys()} for i in range(NumIntegrations)]

	RHSSeqList = []
	IntegratedPDFList = []

	for IntegrationRegion in IntegrationRegions:

		if len(RHSTerms) == 1:
			
			if len(LHSTerms) == 1:
				RHSseq, IntegratedPDF = Integrate1D_Marginalize1D(
					ConditionString, LHSTerms, RHSTerms, LHSDimensions, RHSDimensions,
					DataDict, ConditionalDist, IntegrationRegion)
			elif len(LHSTerms) == 2:
				RHSseq, IntegratedPDF = Integrate2D_Marginalize1D(
					ConditionString, LHSTerms, RHSTerms, LHSDimensions, RHSDimensions,
					DataDict, ConditionalDist, IntegrationRegion)

		elif len(RHSTerms) == 2:

			if len(LHSTerms) == 1:
				RHSseq, IntegratedPDF = Integrate1D_Marginalize2D(
					ConditionString, LHSTerms, RHSTerms, LHSDimensions, RHSDimensions,
					DataDict, ConditionalDist, IntegrationRegion)
			elif len(LHSTerms) == 2:
				RHSseq, IntegratedPDF = Integrate2D_Marginalize2D(
					ConditionString, LHSTerms, RHSTerms, LHSDimensions, RHSDimensions,
					DataDict, ConditionalDist, IntegrationRegion)

		RHSSeqList.append(RHSseq)
		IntegratedPDFList.append(IntegratedPDF)

	return RHSSeqList, IntegratedPDFList


# ConditionalDist, MeanPDF, VariancePDF = CalculateMarginalizeDist(
	# ConditionString=aConditionString, 
	# DataDict=DataDict, weights=weights, 
	# deg_per_dim=deg_per_dim, 
	# JointDist=JointDist
	# )


ConditionString = 'r|p,stm'
IntegrationBounds = {
	'r':[[np.log10(0), np.log10(2)], [np.log10(2), np.log10(4)], 
	[np.log10(4), np.log10(16.4)], [np.log10(0), np.log10(16.4)]], 
	'p':[[np.log10(0.466), np.log10(36.23)], [np.log10(0.466), np.log10(36.23)], 
	[np.log10(0.466), np.log10(36.23)], [np.log10(0.466), np.log10(36.23)]]}
# IntegrationBounds = {'r':[np.log10(0), np.log10(2)]}

RHSSeqList, IntegratedPDFList = IntegrateConditionalDistribution(
	ConditionString, IntegrationBounds,
	DataDict, weights, deg_per_dim, JointDist)

i = 3
plt.figure()
plt.imshow(IntegratedPDFList[i], 
	extent=(RHSSeqList[i][0].min(), RHSSeqList[i][0].max(), RHSSeqList[i][1].min(), RHSSeqList[i][1].max()),
	aspect='auto')
plt.title("Radius 0-17 R_E")
plt.colorbar()


XTicks = np.linspace(RHSSeqList[i][0].min(), RHSSeqList[i][0].max(), 5)
# XTicks = np.log10(np.array([0.3, 1, 3, 10, 30, 100, 300]))
YTicks = np.linspace(RHSSeqList[i][1].min(), RHSSeqList[i][1].max(), 5)
# YTicks = np.log10(np.array([1, 3, 5, 10]))


XLabels = np.round(10**XTicks, 1)
YLabels = np.round(10**YTicks, 2)

plt.xticks(XTicks, XLabels)
plt.yticks(YTicks, YLabels)
plt.xlabel(DataDict['ndim_label'][1])
plt.ylabel(DataDict['ndim_label'][2])
plt.show(block=False)
	
################################################################
"""
PDFIntegralJup = np.zeros(50)
PDFIntegralNep = np.zeros(50)
PDFIntegralRocky = np.zeros(50)

for i in range(50):

	print(10**(zseq[i]))
	MeasurementDict = {RHSTerms[0]:[[zseq[i]], [np.nan]]}#, 'p':[[np.log10(30)],[np.nan]]}


	ConditionalDist, MeanPDF, VariancePDF = calculate_conditional_distribution(
		ConditionString, DataDict, weights, deg_per_dim,
		JointDist, MeasurementDict)
		
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

"""
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
