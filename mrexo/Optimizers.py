from scipy.optimize import fmin_slsqp, minimize
import numpy as np
import datetime
from scipy import sparse
from .utils_nd import _logging
import os


def LogLikelihood(C_pdf, w, n, sparse=False):
	if sparse:
		return np.sum(np.log((w.transpose() * C_pdf).toarray()))
	else:
		return np.sum(np.log(np.matmul(w, C_pdf)))

def SLSQP_optimizer(C_pdf, deg, verbose, save_path):

    # Ensure that the weights always sum up to 1.
    def eqn(w):
        return np.sum(w) - 1

    # Function input to optimizer
    def fn1(w):
        a = - np.sum(np.log(np.matmul(w,C_pdf))) 
        return a

    # Define a list of lists of bounds
    bounds = [[0,1]]*(deg-2)**2
    # Initial value for weights
    x0 = np.repeat(1./((deg-2)**2),(deg-2)**2)

    # Run optimization to find optimum value for each degree (weights). These are the coefficients for the beta densities being used as a linear basis.
    opt_result = fmin_slsqp(fn1, x0, bounds=bounds, f_eqcons=eqn, iter=250, full_output=True, iprint=1,
                            epsilon=1e-5, acc=1e-5)
    message = '\nOptimization run finished at {}, with {} iterations. Exit Code = {}\n\n'.format(datetime.datetime.now(),
            opt_result[2], opt_result[3], opt_result[4])
    _ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)


    unpadded_weight = opt_result[0]
    n_log_lik = opt_result[1]

    return unpadded_weight, n_log_lik


# Ndim - 20201123
def optimizer(C_pdf, deg_per_dim, verbose, save_path, MaxIter=500, rtol=1e-3):
    """
    Using MM algorithm
    INPUTS:
        C_pdf: 2 dimensional matrix. The nominal shape is ((d1-2)*(d2-2), n)
            Assuming that each dimension has a different number of degrees.
        deg_per_dim: A vector or 1D array with degrees corresponding to each dimension.
            Example: deg_per_dim = [5, 7, 9] # Here the 1st, 2nd and 3rd dimensions have
            5, 7, and 9 dimensions respectively.

    20201123 - Adjusted for n dimensions
    """

    C_pdf_sparse = C_pdf
    C_pdf_sparse = sparse.csr_matrix(C_pdf)

    ReducedDegs = np.array(deg_per_dim) - 2
    n = np.shape(C_pdf_sparse)[1] # Sample size

    DegProduct = np.product(ReducedDegs)

    # Initial value for weights
    w = np.ones(DegProduct)/DegProduct
    # w_sparse = sparse.csr_matrix(w)
    w_sparse = sparse.csr_matrix(w[:, None])

    # w_final = np.zeros(np.shape(x0))

    FractionalError = np.ones(MaxIter)
    loglike = np.zeros(MaxIter)
    #loglike_old = np.zeros(MaxIter)

    t = 1

    while np.abs(FractionalError[t-1]) > rtol:
        #TempMatrix =  C_pdf * w[:, None]
        TempMatrix = C_pdf_sparse.multiply(w_sparse)
        IntMatrix = TempMatrix / np.sum(TempMatrix, axis=0)
        w_sparse = sparse.csr_matrix(np.mean(IntMatrix, axis=1))

        #loglike_old[t] = LogLikelihood(C_pdf, w, n)
        loglike[t] = LogLikelihood(C_pdf_sparse, w_sparse, n, sparse=True)
        FractionalError[t] = (loglike[t] - loglike[t-1])/np.abs(loglike[t-1])

        t+=1

        if t == MaxIter:
            break

    message = "Optimization run finished at {}, with {} iterations.\nSum of weights = {} \
        \nLogLikelihood = {}, Fractional Error = {}\n\n".format(datetime.datetime.now(), t, np.sum(w), loglike[t-1], FractionalError[t-1])
    _ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)


    return w_sparse.todense().flatten(), loglike[np.nonzero(loglike)][-1]
