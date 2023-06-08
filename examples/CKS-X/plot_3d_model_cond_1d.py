import os, sys
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec #for specifying plot attributes
import numpy as np
import glob

sys.path.append("/Users/hematthi/Documents/MRExo/mrexo/")
from mrexo.mle_utils_nd import calculate_conditional_distribution, NumericalIntegrate2D





savefigures = False

##### To generate conditional distributions (2-d images) based on a 3-d model marginalized over 1-d:

# Load the model:
#run_path = 'CKS-X_period_radius_stmass_aic'
run_path = 'CKS-X_reduced_period_radius_stmass_deg30'
#run_path = 'CKS-X_flux_radius_stmass'

deg_per_dim = np.loadtxt(os.path.join(run_path, 'output', 'deg_per_dim.txt')).astype(int)
DataDict = np.load(os.path.join(run_path, 'input', 'DataDict.npy'), allow_pickle=True).item()
DataSequences = np.loadtxt(os.path.join(run_path, 'output', 'other_data_products', 'DataSequences.txt'))
weights = np.loadtxt(os.path.join(run_path, 'output', 'weights.txt'))
JointDist = np.load(os.path.join(run_path, 'output', 'JointDist.npy'), allow_pickle=True)

deg_vec_per_dim = [np.arange(1, deg+1) for deg in deg_per_dim]
ndim = DataDict['ndim']

# Define the dimensions to be conditioned (must have the same strings as defined in the run for the model!):
#ConditionString = 'S,Rp|Mstar'
ConditionString = 'P,Rp|Mstar'

ConditionName = ConditionString.replace('|', '_cond_').replace(',', '_')
PlotFolder = os.path.join(run_path, ConditionName)

Condition = ConditionString.split('|')
LHSTerms = Condition[0].split(',')
RHSTerms = Condition[1].split(',')

LHSDimensions = np.arange(ndim)[np.isin(DataDict['ndim_char'], LHSTerms)] #np.array([np.arange(DataDict['ndim'])[np.isin(DataDict['ndim_char'], l)][0] for l in LHSTerms])
RHSDimensions = np.arange(ndim)[np.isin(DataDict['ndim_char'], RHSTerms)] #np.array([np.arange(DataDict['ndim'])[np.isin(DataDict['ndim_char'], r)][0] for r in RHSTerms])

idx, idy = LHSDimensions[0], LHSDimensions[1]
idz = RHSDimensions[0]

xseq = DataDict['DataSequence'][idx]
yseq = DataDict['DataSequence'][idy]
zseq = DataDict['DataSequence'][idz]

# To plot the conditional distribution at a number of points:
zcond = zseq[::2]
#zcond = np.log10([0.6, 0.8, 1., 1.2])
for k,logz in enumerate(zcond):
    zval = 10.**logz
    print('k=%s: %s = %s' % (k, RHSTerms[0], zval))

    MeasurementDict = {RHSTerms[0]:[[logz], [[np.nan, np.nan]]]}

    ConditionalDist, MeanPDF, VariancePDF = calculate_conditional_distribution(ConditionString, DataDict, weights, deg_per_dim,
		JointDist, MeasurementDict)

    _ = NumericalIntegrate2D(xseq, yseq, ConditionalDist[0], [xseq.min(), xseq.max()], [yseq.min(), yseq.max()]) # total integrated probability; should be close to 1
    print('Total probability (integrated): ', _)

    fig = plt.figure(figsize=(8,6))
    plot = GridSpec(1,1,left=0.15,bottom=0.15,right=0.9,top=0.9,wspace=0,hspace=0)
    ax = plt.subplot(plot[0,0])
    im = plt.imshow(ConditionalDist[0], extent=(xseq.min(), xseq.max(), yseq.min(), yseq.max()), aspect='auto', origin='lower');
    #plt.plot(np.log10(DataDict['ndim_data'][idx]), np.log10(DataDict['ndim_data'][idy]),  'k.') # also plot all the data
    plt.title(DataDict['ndim_label'][idz] + ' = {:.2f}'.format(zval), fontsize=20)
    plt.xlabel(DataDict['ndim_label'][idx], fontsize=20)
    plt.ylabel(DataDict['ndim_label'][idy], fontsize=20)

    ax.tick_params(axis='both', labelsize=16)
    plt.xlim(DataDict['ndim_bounds'][idx]) # [::-1] to reverse x-direction, e.g. for flux
    plt.ylim(DataDict['ndim_bounds'][idy])
    #plt.tight_layout()

    xticks = np.linspace(xseq.min(), xseq.max(), 5)
    yticks = np.linspace(yseq.min(), yseq.max(), 5)
    plt.xticks(xticks, np.round(10**xticks, 1))
    plt.yticks(yticks, np.round(10**yticks, 2))
    cbar = fig.colorbar(im, fraction=0.1, pad=0.04) #, label='Probability density'
    #cbar = fig.colorbar(im, ticks=[np.min(ConditionalDist[0]), np.max(ConditionalDist[0])], fraction=0.04, pad=0.04)
    #cbar.ax.set_yticklabels(['Min', 'Max'])
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Probability density', fontsize=16)
    #plt.tight_layout()
    plt.show(block=False)

    if savefigures:
        #plt.savefig(os.path.join(PlotFolder, ConditionName + '_{:.2f}.png'.format(zval)))
        plt.savefig(os.path.join(PlotFolder, ConditionName + '_%s.png' % k))
    #plt.close("all")



##### To create a movie with the images?
"""
import imageio

ListofPlots = glob.glob(os.path.join(PlotFolder, '3*.png'))
ListofPlots.sort(key=os.path.getmtime)

writer = imageio.get_writer(os.path.join(PlotFolder, ConditionName+'.mp4'), fps=2) # DOES NOT WORK (pip install imageio[ffmpeg] also doesn't work)

for im in ListofPlots:
		#print(order, im)
		writer.append_data(imageio.imread(os.path.join(PlotFolder, im)))
writer.close()
"""
