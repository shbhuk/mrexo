import numpy as np
import os
from multiprocessing import current_process
import sys
if sys.version_info.major==3:
	from functools import lru_cache
else:
	from functools32 import lru_cache

def _save_dictionary(dictionary, output_location,
	bootstrap=False):

	"""
	Need to update
	Save the keys in the dictionary as separate data .txt files.
	"""
	aux_output_location = os.path.join(output_location, 'other_data_products')

	if not bootstrap:
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

def GiveDegreeCandidates(degree_max, n, ndim, ncandidates=10):
	"""
	INPUTS:
		degree_max = A np.array with number of elements equal to number of degrees, 
			with each element corresponding to the maximum degree for each dimension.
			Or else an integer
		ndim = Number of dimensions
		n = Size of dataset
	"""
	
	if type(degree_max) == int:
		degree_candidates = np.array([np.linspace(5, degree_max, ncandidates, dtype=int) for i in range(ndim)])
	else:
		degree_candidates =  np.array([np.linspace(5, d, ncandidates, dtype=int) for d in degree_max])

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
