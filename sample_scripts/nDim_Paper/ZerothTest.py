from memory_profiler import profile
import numpy as np
from scipy import sparse 

@profile
def RandomNumpyMatrix():
	np.random.seed(1728)
	bigmatrix = np.random.normal(size=(3000, 2000))
	bigmatrix[bigmatrix < 0.5] = 0
	return bigmatrix


@profile
def RandomSparseMatrix():
	np.random.seed(1728)
	bigmatrix = np.random.normal(size=(3000, 2000))
	bigmatrix[bigmatrix < 0.5] = 0
	sp = sparse.csr_matrix(bigmatrix)
	# sr = sparse.random(3000, 2000, density=0.25)
	return sp


_ = RandomNumpyMatrix()
_ = RandomSparseMatrix()
