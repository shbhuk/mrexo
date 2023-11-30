import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import glob, os

from mrexo.mle_utils_nd import calculate_conditional_distribution, NumericalIntegrate2D

location = os.path.dirname(os.path.abspath(__file__))
#location = "/data/skanodia/work/mrexo/sample_scripts/MdwarfRuns"
#print(location)

matplotlib.rcParams['xtick.labelsize'] = 25
matplotlib.rcParams['ytick.labelsize'] = 25
cmap = matplotlib.cm.viridis
cmap = matplotlib.cm.Spectral

def CleanInput(a):
	if type(a) == list: return np.array(a)
	elif type(a) is not np.ndarray: return np.array([a])
	else: return a

def Mdwarf_InferPlMass_FromPlRadiusInsolStMass(
	pl_rade, pl_insol, st_mass):
	"""
	Inputs:
		pl_rade = Planetary radius in Earth radius. Specify as numpy array
		pl_insol = Planetary insolation in Earth units. Specify as numpy array
		st_mass = Stellar mass in solar mass. Specify as numpy array
	Outputs:
		MeanPDF
	"""
	ConditionString = 'm|r,insol,stm'
	ConditionName = '4D_'+ConditionString.replace('|', '_').replace(',', '_')
	
	RunName = r"MdwarfPlanets_4D_MRSStM_20231102"

	save_path = os.path.join(location, RunName)

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
	

	xdata = DataDict['ndim_data'][RHSDimensions[0]]
	ydata = DataDict['ndim_data'][RHSDimensions[1]]
	zdata = DataDict['ndim_data'][RHSDimensions[2]]

	Bounds = DataDict['ndim_bounds']
	
	x = np.log10(CleanInput(pl_rade))
	y = np.log10(CleanInput(pl_insol))
	z = np.log10(CleanInput(st_mass))

	if not np.all(x > Bounds[0][0]) & np.all(x < Bounds[0][1]): print("Input pl_rade is out of bounds = ", *10**Bounds[0]); return np.nan
	if not np.all(y > Bounds[2][0]) & np.all(y < Bounds[2][1]): print("Input pl_insol is out of bounds = ", *10**Bounds[2]); return np.nan
	if not np.all(z > Bounds[3][0]) & np.all(z < Bounds[3][1]): print("Input st_mass is out of bounds = ", *10**Bounds[3]); return np.nan

	CombinedQuery = np.rollaxis(np.array(np.meshgrid(x, y, z)), 0, 4).reshape(len(x)*len(y)*len(z),3)

	MeasurementDict=	{
				'r':[CombinedQuery[:,0], [[np.nan, np.nan]]*len(CombinedQuery)], 
				'insol':[CombinedQuery[:,1], [[np.nan, np.nan]]*len(CombinedQuery)],
				'stm':[CombinedQuery[:,2], [[np.nan, np.nan]]*len(CombinedQuery)]
	}

	ConditionalDist, MeanPDF, VariancePDF = calculate_conditional_distribution(
		ConditionString, DataDict, weights, deg_per_dim,
		JointDist.T, MeasurementDict)

	MeanPDF = MeanPDF.reshape((len(x), len(y), len(z)))

	return 10**MeanPDF



def Mdwarf_InferPlMass_FromPlRadiusStMass(
	pl_rade, st_mass):
	"""
	Inputs:
		pl_rade = Planetary radius in Earth radius. Specify as numpy array
		st_mass = Stellar mass in solar mass. Specify as numpy array
	Outputs:
		MeanPDF
	"""
 
	ConditionString = 'm|r,stm'
	ConditionName = '3D_'+ConditionString.replace('|', '_').replace(',', '_')
	
	RunName = r"MdwarfPlanets_3D_MRStM_20231102"

	save_path = os.path.join(location, RunName)

	PlotFolder = os.path.join(save_path, ConditionName)
	
	if not os.path.exists(PlotFolder):
		print("3D Plot folder does not exist")
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
	

	xdata = DataDict['ndim_data'][RHSDimensions[0]]
	ydata = DataDict['ndim_data'][RHSDimensions[1]]

	Bounds = DataDict['ndim_bounds']
	
	x = np.log10(CleanInput(pl_rade))
	y = np.log10(CleanInput(st_mass))

	if not np.all(x > Bounds[0][0]) & np.all(x < Bounds[0][1]): print("Input pl_rade is out of bounds = ", *10**Bounds[0]); return np.nan
	if not np.all(y > Bounds[2][0]) & np.all(y < Bounds[2][1]): print("Input st_mass is out of bounds = ", *10**Bounds[2]); return np.nan

	CombinedQuery = np.rollaxis(np.array(np.meshgrid(x, y)), 0, 3).reshape(len(x)*len(y), 2)

	MeasurementDict=	{
				'r':[CombinedQuery[:,0], [[np.nan, np.nan]]*len(CombinedQuery)], 
				'stm':[CombinedQuery[:,1], [[np.nan, np.nan]]*len(CombinedQuery)]
	}

	ConditionalDist, MeanPDF, VariancePDF = calculate_conditional_distribution(
		ConditionString, DataDict, weights, deg_per_dim,
		JointDist.T, MeasurementDict)

	MeanPDF = MeanPDF.reshape((len(x), len(y)))

	return 10**MeanPDF




def Mdwarf_InferPlMass_FromPlRadius(
	pl_rade):
	"""
	Inputs:
		pl_rade = Planetary radius in Earth radius. Specify as numpy array
	Outputs:
		MeanPDF
	"""
 
	ConditionString = 'm|r'
	ConditionName = '2D_'+ConditionString.replace('|', '_').replace(',', '_')
	
	RunName = r"MdwarfPlanets_2D_MR_20231102"

	save_path = os.path.join(location, RunName)

	PlotFolder = os.path.join(save_path, ConditionName)
	
	if not os.path.exists(PlotFolder):
		print("3D Plot folder does not exist")
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
	

	xdata = DataDict['ndim_data'][RHSDimensions[0]]

	Bounds = DataDict['ndim_bounds']
	
	x = np.log10(CleanInput(pl_rade))

	if not np.all(x > Bounds[0][0]) & np.all(x < Bounds[0][1]): print("Input pl_rade is out of bounds = ", *10**Bounds[0]); return np.nan

	MeasurementDict=	{
				'r':[x, [[np.nan, np.nan]]*len(x)]
	}

	ConditionalDist, MeanPDF, VariancePDF = calculate_conditional_distribution(
		ConditionString, DataDict, weights, deg_per_dim,
		JointDist.T, MeasurementDict)

	return 10**MeanPDF

#print(Mdwarf_InferPlMass_FromPlRadiusInsolStMass(pl_rade=10, pl_insol=100, st_mass=0.5))
#print(Mdwarf_InferPlMass_FromPlRadiusInsolStMass(pl_rade=[8, 10, 12], pl_insol=[100, 500], st_mass=[0.4, 0.5, 0.6]))
#print(Mdwarf_InferPlMass_FromPlRadiusStMass(pl_rade=[8, 10, 12],  st_mass=[0.4, 0.5, 0.6]))
#print(Mdwarf_InferPlMass_FromPlRadius(pl_rade = [8, 10, 12]))