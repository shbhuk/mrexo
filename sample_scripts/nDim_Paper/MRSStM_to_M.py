import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib import gridspec
from scipy.interpolate import RectBivariateSpline, UnivariateSpline
from scipy.integrate import simps

import numpy as np
# import imageio
import glob, os
import pandas as pd
from mrexo.mle_utils_nd import calculate_conditional_distribution, NumericalIntegrate2D
import datetime
# import gc # import garbage collector interface

# from pyastrotools.general_tools import SigmaAntiLog10, SigmaLog10
# from pyastrotools.astro_tools import calculate_orbperiod

cmap = matplotlib.cm.YlGnBu

ConditionString = 'm|r,insol,stm'
ConditionName = '4D_'+ConditionString.replace('|', '_').replace(',', '_')

ResultDirectory = r'C:\Users\skanodia\Documents\GitHub\mrexo\sample_scripts\TestRuns\AllPlanet_RpLt4_StMlt1.5_MRSStM_d40_100MC_100BS'
PlotFolder = os.path.join(ResultDirectory, ConditionName)
PlotFolder = os.path.join(ResultDirectory, '4D_m_r_stm_insol')

if not os.path.exists(PlotFolder):
	print("4D Plot folder does not exist")
	os.mkdir(PlotFolder)

deg_per_dim = np.loadtxt(os.path.join(ResultDirectory, 'output', 'deg_per_dim.txt'))
DataDict = np.load(os.path.join(ResultDirectory, 'input', 'DataDict.npy'), allow_pickle=True).item()
JointDist = np.load(os.path.join(ResultDirectory, 'output', 'JointDist.npy'), allow_pickle=True).T
weights = np.genfromtxt(os.path.join(ResultDirectory, 'output', 'weights.txt'))
deg_per_dim = np.loadtxt(os.path.join(ResultDirectory, 'output', 'deg_per_dim.txt')).astype(int)

Condition = ConditionString.split('|')
LHSTerms = Condition[0].split(',')
RHSTerms = Condition[1].split(',')
deg_vec_per_dim = [np.arange(1, deg+1) for deg in deg_per_dim] 

LHSDimensions = np.array([(np.arange(DataDict['ndim'])[np.isin(DataDict['ndim_char'] , l)])[0] for l in LHSTerms])
RHSDimensions = np.array([(np.arange(DataDict['ndim'])[np.isin(DataDict['ndim_char'] , r)])[0] for r in RHSTerms])

x = DataDict['DataSequence'][RHSDimensions[0]] # pl_rade
y = DataDict['DataSequence'][RHSDimensions[1]][::2] # insol
y = np.log10([10, 50, 100, 500])
z = DataDict['DataSequence'][RHSDimensions[2]] # stm
MassAxis = DataDict['DataSequence'][LHSDimensions[0]] # Pl Mass

xdata = DataDict['ndim_data'][RHSDimensions[0]] # r
ydata = DataDict['ndim_data'][RHSDimensions[1]] # insol
zdata = DataDict['ndim_data'][RHSDimensions[2]] # stm

cmap = matplotlib.cm.YlGnBu
# Establish colour range based on variable
# norm = matplotlib.colors.Normalize(vmin=ConsiderRadii.min()/2, vmax=ConsiderRadii.max())

StellarMassArray = np.arange(0.4, 1.35, 0.05)
InsolationArray = 10**np.arange(0, 3, 0.25)
InsolationArray = np.linspace(1, 1000, 20)
# StellarMassArray = np.arange(0.4, 1.35, 0.1)
# InsolationArray = 10**np.arange(0, 3, 0.5)
# InsolationArray = np.array([100, 300])
PlanetRadiusArray = [1.5]

CombinedQuery = np.rollaxis(np.array(np.meshgrid(StellarMassArray, InsolationArray, PlanetRadiusArray)), 0, 3).reshape(len(StellarMassArray)*len(InsolationArray), 3)

MeasurementDict=  {
			'stm':[np.log10(CombinedQuery[:,0]), [[np.nan, np.nan]]*len(CombinedQuery)], 
			'insol':[np.log10(CombinedQuery[:,1]), [[np.nan, np.nan]]*len(CombinedQuery)],
			'r':[np.log10(CombinedQuery[:,2]), [[np.nan, np.nan]]*len(CombinedQuery)],
}

ConditionalDist, MeanPDF, VariancePDF = calculate_conditional_distribution(ConditionString, DataDict, weights, deg_per_dim,
	JointDist.T, MeasurementDict)
MeanMass = np.reshape(MeanPDF, (len(InsolationArray), len(StellarMassArray)))


im = plt.imshow(10**MeanMass, origin='lower', aspect='auto', extent=(StellarMassArray.min(), StellarMassArray.max(), InsolationArray.min(), InsolationArray.max()), interpolation='bicubic', cmap=cmap, vmin=2, vmax=9)
cbar = plt.colorbar(im, pad = 0.05)
cbar.set_label(r"Planetary Mass ($M_{\oplus}$)",  size=25)
cbar.ax.tick_params(labelsize=25)

# _cs2 = plt.contour(10**MeanMass, levels=[4.14869771, 4.44446175, 4.73118858], origin='lower' , extent=(StellarMassArray.min(), StellarMassArray.max(), InsolationArray.min(), InsolationArray.max()))


plt.ylabel("Insolation Flux ($S_{\oplus}$)", fontsize=25)
plt.xlabel(r"Stellar Mass ($M_{\odot}$)", fontsize=25)
plt.title(r"f(m|r=1.5 $R_{\oplus}$, insol, stm)")
plt.yticks(fontsize=30)
plt.xticks(fontsize=30)
plt.tight_layout()
plt.show(block=False)

# plt.savefig(os.path.join(r"C:\Users\skanodia\Documents\GitHub\Paper_ndim_mrexo\Plots", "MRSStM_Rp1.5.pdf"))
