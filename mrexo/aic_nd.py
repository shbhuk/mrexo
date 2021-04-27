# -*- coding: utf-8 -*-
import numpy as np
import os
import pandas as pd
from multiprocessing import Pool
from .mle_utils_nd import MLE_fit, calc_C_matrix
from .utils import _save_dictionary, _logging, GiveDegreeCandidates
import matplotlib.pyplot as plt

def run_aic(DataDict, degree_max, NumCandidates=20, cores=1,
	save_path=os.path.dirname(__file__), verbose=2, abs_tol=1e-8):

	ndim = DataDict['ndim']
	n = DataDict['DataLength']

	degree_candidates = GiveDegreeCandidates(degree_max=degree_max, n=n, ndim=ndim, ncandidates=NumCandidates)

	message = 'Using AIC method to estimate the number of degrees of freedom for the weights. Max candidate = {}\n'.format(degree_candidates.max())
	_ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)
	
	if ndim==2:
		DegreeChosen = RunAIC2D(DataDict=DataDict, 
			degree_candidates=degree_candidates, NumCandidates=NumCandidates, 
			save_path=save_path)
	elif ndim==3:
		DegreeChosen = RunAIC3D(DataDict=DataDict, 
			degree_candidates=degree_candidates, NumCandidates=NumCandidates, 
			save_path=save_path)
	elif ndim==4:
		DegreeChosen = RunAIC4D(DataDict=DataDict, 
			degree_candidates=degree_candidates, NumCandidates=NumCandidates, 
			save_path=save_path)	
	
	message = 'Using AIC to select optimum degrees as = {}\n'.format(DegreeChosen)
	_ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)
	
	
	return DegreeChosen


def MakePlot(Data, Title, degree_candidates):
	"""
	
	"""
	
	plt.close("all")
	fig = plt.figure()
	plt.imshow(Data, extent=[degree_candidates[0].min(), degree_candidates[0].max(), degree_candidates[1].min(), degree_candidates[1].max()], origin='lower')
	plt.title(Title)
	plt.colorbar()
	
	return fig

def RunAIC2D(DataDict, degree_candidates, NumCandidates, save_path):
	
	n = DataDict['DataLength']
	ndim = DataDict['ndim']

	AIC = np.zeros(([NumCandidates]*ndim))
	FI = np.zeros(np.shape(AIC))

	loglike = np.zeros(np.shape(AIC))
	DegProduct = np.zeros(np.shape(AIC))
	NonZero = np.zeros(np.shape(AIC))
	Threshold = np.zeros(np.shape(AIC))
	
	for i in range(0, NumCandidates):
		print("Running AIC:" + str(i))
		for j in range(0, NumCandidates):
			
		
			deg_per_dim = [degree_candidates[0][i], degree_candidates[1][j]]
			deg_product = np.product(deg_per_dim)
			
			output = MLE_fit(DataDict,  deg_per_dim=deg_per_dim,
				save_path=save_path, OutputWeightsOnly=False, CalculateJointDist=False, verbose=1)
				
			Weights = output['Weights']
				
			loglike[i,j] = output['loglike']
			DegProduct[i,j] = deg_product
			AIC[i,j] = output['aic']
			FI[i,j] = output['fi']

			NonZero[i,j] = len(np.nonzero(Weights)[0])
			Threshold[i,j] = len(Weights[Weights > 1e-8])
		
	MinAICIndexFlat = np.argmin(AIC)
	MinAICIndex = np.unravel_index(MinAICIndexFlat, np.shape(AIC))
	DegreeChosen = np.array([degree_candidates[i][MinAICIndex[i]] for i in range(ndim)], dtype=int)
			
	np.savetxt(os.path.join(save_path, 'output', 'other_data_products', 'degree_candidates.txt'), degree_candidates)
	np.save(os.path.join(save_path, 'output', 'other_data_products', 'AIC.npy'), AIC)
	np.save(os.path.join(save_path, 'output', 'other_data_products', 'loglike.npy'), loglike)
	np.save(os.path.join(save_path, 'output', 'other_data_products', 'FI.npy'), FI)
	np.save(os.path.join(save_path, 'output', 'other_data_products', 'DegProduct.npy'), DegProduct)
	np.save(os.path.join(save_path, 'output', 'other_data_products', 'NonZero.npy'), NonZero)
	
			
	fig = MakePlot(loglike, Title='loglike', degree_candidates=degree_candidates)
	fig.savefig(os.path.join(save_path, 'output', 'other_data_products', 'loglike.png'))
	
	fig = MakePlot(AIC, Title='AIC', degree_candidates=degree_candidates)
	fig.savefig(os.path.join(save_path, 'output', 'other_data_products', 'AIC.png'))

	fig = MakePlot(FI, Title='FI', degree_candidates=degree_candidates)
	fig.savefig(os.path.join(save_path, 'output', 'other_data_products', 'FI.png'))

	fig = MakePlot(DegProduct, Title='DegProduct', degree_candidates=degree_candidates)
	fig.savefig(os.path.join(save_path, 'output', 'other_data_products', 'DegProduct.png'))

	fig = MakePlot(2*(DegProduct/n - loglike), Title="AIC = 2*(DegProduct/n - LogLike)", degree_candidates=degree_candidates)
	fig.savefig(os.path.join(save_path, 'output', 'other_data_products', 'DegProducts_n_AIC.png'))

	fig = MakePlot(2*(NonZero/n - loglike), Title="AIC = 2*(NonZero/n - LogLike)", degree_candidates=degree_candidates)
	fig.savefig(os.path.join(save_path, 'output', 'other_data_products', 'NonZero_n_AIC.png'))

	fig = MakePlot(2*(FI - loglike), Title="AIC = 2*(FI - LogLike)", degree_candidates=degree_candidates)
	fig.savefig(os.path.join(save_path, 'output', 'other_data_products', 'FI_AIC.png'))

	fig = MakePlot(2*(FI/n - loglike), Title="AIC = 2*(FI/n - LogLike)", degree_candidates=degree_candidates)
	fig.savefig(os.path.join(save_path, 'output', 'other_data_products', 'FI_n_AIC.png'))
	
	return DegreeChosen
	
	
	
def RunAIC3D(DataDict, degree_candidates, NumCandidates, save_path):
	
	n = DataDict['DataLength']
	ndim = DataDict['ndim']

	AIC = np.zeros(([NumCandidates]*ndim))
	FI = np.zeros(np.shape(AIC))

	loglike = np.zeros(np.shape(AIC))
	DegProduct = np.zeros(np.shape(AIC))
	NonZero = np.zeros(np.shape(AIC))
	Threshold = np.zeros(np.shape(AIC))
	
	for i in range(0, NumCandidates):
		for j in range(0, NumCandidates):
			print("Running AIC:" + str(i)+','+str(j))
			for k in range(0, NumCandidates):
				
				deg_per_dim = [degree_candidates[0][i], degree_candidates[1][j], degree_candidates[2][k]]
				deg_product = np.product(deg_per_dim)
				
				output = MLE_fit(DataDict,  deg_per_dim=deg_per_dim,
					save_path=save_path, OutputWeightsOnly=False, CalculateJointDist=False, verbose=1)
					
				Weights = output['Weights']
					
				loglike[i,j,k] = output['loglike']
				DegProduct[i,j,k] = deg_product
				AIC[i,j,k] = output['aic']
				FI[i,j,k] = output['fi']

				NonZero[i,j,k] = len(np.nonzero(Weights)[0])
				Threshold[i,j,k] = len(Weights[Weights > 1e-8])
		
	MinAICIndexFlat = np.argmin(AIC)
	MinAICIndex = np.unravel_index(MinAICIndexFlat, np.shape(AIC))
	DegreeChosen = np.array([degree_candidates[i][MinAICIndex[i]] for i in range(ndim)], dtype=int)

	np.savetxt(os.path.join(save_path, 'output', 'other_data_products', 'degree_candidates.txt'), degree_candidates)
	np.save(os.path.join(save_path, 'output', 'other_data_products', 'AIC.npy'), AIC)
	np.save(os.path.join(save_path, 'output', 'other_data_products', 'loglike.npy'), loglike)
	np.save(os.path.join(save_path, 'output', 'other_data_products', 'FI.npy'), FI)
	np.save(os.path.join(save_path, 'output', 'other_data_products', 'DegProduct.npy'), DegProduct)
	np.save(os.path.join(save_path, 'output', 'other_data_products', 'NonZero.npy'), NonZero)
	
			
	return DegreeChosen
	

def RunAIC4D(DataDict, degree_candidates, NumCandidates, save_path):
	
	n = DataDict['DataLength']
	ndim = DataDict['ndim']

	AIC = np.zeros(([NumCandidates]*ndim))
	FI = np.zeros(np.shape(AIC))

	loglike = np.zeros(np.shape(AIC))
	DegProduct = np.zeros(np.shape(AIC))
	NonZero = np.zeros(np.shape(AIC))
	Threshold = np.zeros(np.shape(AIC))
	
	for i in range(0, NumCandidates):
		for j in range(0, NumCandidates):
			for k in range(0, NumCandidates):
				print("Running AIC: {}{}{}".format(i,j,k))
				for l in range(0, NumCandidates):
						
					deg_per_dim = [degree_candidates[0][i], degree_candidates[1][j], degree_candidates[2][k], degree_candidates[3][l]]
					deg_product = np.product(deg_per_dim)
					
					output = MLE_fit(DataDict,  deg_per_dim=deg_per_dim,
						save_path=save_path, OutputWeightsOnly=False, CalculateJointDist=False, verbose=1)
						
					Weights = output['Weights']
						
					loglike[i,j,k,l] = output['loglike']
					DegProduct[i,j,k,l] = deg_product
					AIC[i,j,k,l] = output['aic']
					FI[i,j,k,l] = output['fi']

					NonZero[i,j,k,l] = len(np.nonzero(Weights)[0])
					Threshold[i,j,k,l] = len(Weights[Weights > 1e-8])
			
	MinAICIndexFlat = np.argmin(AIC)
	MinAICIndex = np.unravel_index(MinAICIndexFlat, np.shape(AIC))
	DegreeChosen = np.array([degree_candidates[i][MinAICIndex[i]] for i in range(ndim)], dtype=int)

	np.savetxt(os.path.join(save_path, 'output', 'other_data_products', 'degree_candidates.txt'), degree_candidates)
	np.save(os.path.join(save_path, 'output', 'other_data_products', 'AIC.npy'), AIC)
	np.save(os.path.join(save_path, 'output', 'other_data_products', 'loglike.npy'), loglike)
	np.save(os.path.join(save_path, 'output', 'other_data_products', 'FI.npy'), FI)
	np.save(os.path.join(save_path, 'output', 'other_data_products', 'DegProduct.npy'), DegProduct)
	np.save(os.path.join(save_path, 'output', 'other_data_products', 'NonZero.npy'), NonZero)
	
			
	return DegreeChosen	
