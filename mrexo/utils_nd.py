import numpy as np
import os
from multiprocessing import current_process
import sys
if sys.version_info.major==3:
	from functools import lru_cache
else:
	from functools32 import lru_cache

def _save_dictionary(dictionary, output_location,
	NumBootstrap=False, NumMonteCarlo=False):

	"""
	Need to update
	Save the keys in the dictionary as separate data .txt files.
	"""
	aux_output_location = os.path.join(output_location, 'other_data_products')


	if NumMonteCarlo is not False:
		unpadded_weights = dictionary['UnpaddedWeights']
		weights = dictionary['Weights']
		JointDist = dictionary['JointDist']

		np.savetxt(os.path.join(output_location,'weights_MCSim{}.txt'.format(str(NumMonteCarlo))),weights, comments='#', header='Weights for Beta densities  from Monte-Carlo Sim # = {}'.format(str(NumMonteCarlo)))
		np.savetxt(os.path.join(output_location,'unpadded_weights_MCSim{}.txt'.format(str(NumMonteCarlo))),unpadded_weights, comments='#', header='Unpadded weights for Beta densities from Monte-Carlo Sim # = {}'.format(str(NumMonteCarlo)))
		np.save(os.path.join(output_location,'JointDist_MCSim{}.npy'.format(str(NumMonteCarlo))), JointDist)
	
	elif NumBootstrap is not False:
		unpadded_weights = dictionary['UnpaddedWeights']
		weights = dictionary['Weights']
		JointDist = dictionary['JointDist']

		np.savetxt(os.path.join(output_location,'weights_BSSim{}.txt'.format(str(NumBootstrap))),weights, comments='#', header='Weights for Beta densities  from Bootstrap # = {}'.format(str(NumBootstrap)))
		np.savetxt(os.path.join(output_location,'unpadded_weights_BSSim{}.txt'.format(str(NumBootstrap))),unpadded_weights, comments='#', header='Unpadded weights for Beta densities from Bootstrap # = {}'.format(str(NumBootstrap)))
		np.save(os.path.join(output_location,'JointDist_BSSim{}.npy'.format(str(NumBootstrap))), JointDist)

	else:
		unpadded_weights = dictionary['UnpaddedWeights']
		weights = dictionary['Weights']
		deg_per_dim = dictionary['deg_per_dim']
		DataSequences = dictionary['DataSequence']
		JointDist = dictionary['JointDist']

		np.savetxt(os.path.join(output_location,'weights.txt'),weights, comments='#', header='Weights for Beta densities from initial fitting w/o bootstrap')
		np.savetxt(os.path.join(output_location,'unpadded_weights.txt'),unpadded_weights, comments='#', header='Unpadded weights for Beta densities from initial fitting w/o bootstrap')
		np.savetxt(os.path.join(output_location,'deg_per_dim.txt'), deg_per_dim, comments='#', header='Degrees per dimensions')
		np.save(os.path.join(output_location,'JointDist.npy'), JointDist)
		np.savetxt(os.path.join(aux_output_location,'DataSequences.txt'), DataSequences, comments='#', header='Data Sequence for each dimensions')
	
def GiveDegreeCandidates(degree_max, ndim, ncandidates=10):
    """
    Create a vector of degree candidates in each dimension.
    
    Parameters
    ----------
    degree_max : int or array[int]
        The maximum degree to be considered (if an integer), or the maximum degree to be considered in each dimension (if an array of integers).
    ndim : int
        The number of dimensions.
    ncandidates : int, default=10
        The number of degree candidates to consider.
    
    Returns
    -------
    degree_candidates : array[int]
        A 2D array containing the vector of degree candidates in each dimension.
    """
	
	if type(degree_max) == int:
		degree_candidates = np.array([np.linspace(10, degree_max, ncandidates, dtype=int) for i in range(ndim)])
	else:
		degree_candidates =  np.array([np.linspace(10, d, ncandidates, dtype=int) for d in degree_max])

	return degree_candidates

@lru_cache(maxsize=200)
def _load_lookup_table(f_path):
	"""
	Load the lookup table interpolate object and pass the object.
	INPUT:
		f_path : Entire file path for the .npy interpolated file.
	OUTPUT:
		lookup_inter : Interpolated lookup table (.npy) object.

	"""

	lookup_inter = np.load(f_path, encoding = 'bytes', allow_pickle=True).item()
	print('Loaded lookup table from {}'.format(f_path))

	return lookup_inter


def _logging(message, filepath, verbose, append=True):
	"""

	"""

	message = str(current_process().pid)+":"+message

	if append:
		action="a"
	else:
		action="w"

	if verbose==1:
		with open(os.path.join(filepath,'log_file.txt'),action) as f:
			f.write('Using core '+message)
	elif verbose==2:
		with open(os.path.join(filepath,'log_file.txt'),action) as f:
			f.write('Using core '+message)
		print('Using core '+message)

	return 1


def MakePlot(Data, Title, degree_candidates, Interpolate=False, AddContour=False):
    """
    Plot a 2D heat-map as a function of the degree candidates in each dimension.
    
    Parameters
    ----------
    Data : array[float]
        The 2-d array to plot.
    Title : str
        The title of the figure.
    degree_candidates : array[int] or list[array[int]]
        The array or list containing the vector of degree candidates in each dimension.
    Interpolate : bool, default=False
        Whether to interpolate the heat-map from ``Data``.
    AddContour : bool, default=False
        Whether to add contour levels to the plot.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot.
    """
    
    plt.close("all")
    fig = plt.figure()
    if Interpolate:
        im = plt.imshow(Data, extent=[degree_candidates[0].min(), degree_candidates[0].max(), degree_candidates[1].min(), degree_candidates[1].max()], origin='lower', interpolation='bicubic')
    else:
        im = plt.imshow(Data, extent=[degree_candidates[0].min(), degree_candidates[0].max(), degree_candidates[1].min(), degree_candidates[1].max()], origin='lower')

    if AddContour:
        contours = plt.contour(degree_candidates[0], degree_candidates[1], Data, 20, colors='black')
        plt.clabel(contours, inline=1, fontsize=10)

    plt.title(Title)
    plt.colorbar(im)
    plt.xlabel("Degrees ($d_1$)")
    plt.ylabel("Degrees ($d_2$)")
    
    return fig


def FlattenGrid(Inputs, ndim):
    """
    Create a flattened mesh of ``Inputs``.
    
    Parameters
    ----------
    Inputs : list[list[float]] or array[float]
        The input list/array of lists/arrays or 2D array.
    ndim : int
        The number of dimensions in (length of) ``Inputs``.
    
    Returns
    -------
    FlattenedMesh : list[list[float]]
        The flattened list of mesh points; see example below.
    
    Examples
    --------
    >>> Inputs = [[1,2,3,4], [1,2,3,4], [1,2,3,4]] # 3 dimensions with 4 points
    >>> FlattenGrid(Inputs, 3)
    [[1,1,1], [1,1,2], [1,1,3], [1,1,4], [2,1,1], [2,1,2], [2,1,3], ..., [4,4,3], [4,4,4]]
    """
    # TODO: can just calculate 'ndim' from the Inputs (e.g., 'ndim = len(Inputs)') instead of passing as a parameter?
    
    Mesh = np.meshgrid(*Inputs)
    i_flat = [Mesh[i].flatten() for i in range(ndim)]
    FlattenedMesh = [[(ix[i]) for ix in i_flat] for i in range(len(i_flat[0]))]
    
    return FlattenedMesh
