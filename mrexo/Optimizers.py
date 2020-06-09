from scipy.optimize import fmin_slsqp, minimize
import numpy as np
import datetime
from mrexo.utils import _logging
import os


def LogLikelihood(Cpdf, w, n):
    return np.sum(np.log(np.matmul(w,Cpdf)))/n

def SLSQP_optimizer(C_pdf, deg, verbose, save_path):

    # Ensure that the weights always sum up to 1.
    def eqn(w):
        return np.sum(w) - 1

    # Function input to optimizer
    def fn1(w):
        a = - np.sum(np.log(np.matmul(w,C_pdf))) / n
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

def optimizer(C_pdf, deg, verbose, save_path, MaxIter=500, rtol=1e-3):
    """
    Using MM algorithm
    """

    ReducedDeg = deg-2
    n = np.shape(C_pdf)[1] # Sample size

    # Initial value for weights
    w = np.repeat(1./(ReducedDeg**2),ReducedDeg**2)

    # w_final = np.zeros(np.shape(x0))

    FractionalError = np.ones(MaxIter)
    loglike = np.zeros(MaxIter)

    t = 1

    while np.abs(FractionalError[t-1]) > rtol:
        TempMatrix =  C_pdf * w[:, None]
        IntMatrix = TempMatrix / np.sum(TempMatrix, axis=0)
        w = np.mean(IntMatrix, axis=1)

        loglike[t] = LogLikelihood(C_pdf, w, n)
        FractionalError[t] = (loglike[t] - loglike[t-1])/np.abs(loglike[t-1])

        t+=1

        if t == MaxIter:
            break

    message = "Optimization run finished at {}, with {} iterations.\nSum of weights = {} \
        \nLogLikelihood = {}, Fractional Error = {}\n\n".format(datetime.datetime.now(), t, np.sum(w), loglike[t-1], FractionalError[t-1])
    _ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)


    return w, loglike[np.nonzero(loglike)][-1]



Bleh = r"""



deg = 30
ReducedDeg = deg-2

'''
randn = "-0.342"

# Existing Cpdf from 34 planet sample
Cpdf = np.loadtxt(r"C:\Users\shbhu\Documents\Git\mrexo\sample_scripts\Mdwarfs_20200520NewOptTrial\output\other_data_products\C_pdf.txt")
Cpdf = np.loadtxt(r"C:\Users\shbhu\Documents\Git\mrexo\sample_scripts\Mdwarfs_20200520NewOptTrial\output\other_data_products\Cpdf"+randn+".txt")

# Weights from existing optimizer - SLSQP
OldWeights = np.loadtxt(r"C:\Users\shbhu\Documents\Git\mrexo\sample_scripts\Mdwarfs_20200520NewOptTrial\output\other_data_products\IntermediateWeight.txt")
OldWeights = np.loadtxt(r"C:\Users\shbhu\Documents\Git\mrexo\sample_scripts\Mdwarfs_20200520NewOptTrial\output\other_data_products\IntermediateWeight"+randn+".txt")
OldWeights = np.reshape(np.reshape(OldWeights, (deg, deg))[1:-1,1:-1], ReducedDeg**2)
'''


Folder = r"C:\Users\shbhu\Documents\Git\MREx_julia\examples"
ResultDir = os.path.join(Folder, 'MR_size24_with_outputs', 'output', 'other_data_products')
ResultDir = os.path.join(Folder, 'MR_size127_with_outputs', 'output', 'other_data_products')
ResultDir = os.path.join(Folder, 'MR_size800_with_outputs_deg55', 'output', 'other_data_products')
ResultDir = os.path.join(Folder, 'PR_size800_with_outputs_deg55', 'output', 'other_data_products')

Cpdf = np.loadtxt(os.path.join(ResultDir, 'C_pdf.csv'), delimiter=',', skiprows=1)
# OldWeights = np.loadtxt(os.path.join(ResultDir, 'unpadded_weight.csv'), delimiter=',', skiprows=1)
OldWeights =np.repeat(1./(53**2),53**2)
ReducedDeg = int(np.sqrt(np.shape(OldWeights)))

def LogLikelihood(Cpdf, w, n):
    return np.sum(np.log(np.matmul(w,Cpdf)))/n # Divide sum by n for stability when n gets large

OldLogLike = LogLikelihood(Cpdf, OldWeights, n)
print(OldLogLike)


# Initial value for weights
x0 = np.repeat(1./(ReducedDeg**2),ReducedDeg**2)

w = x0
w_final = np.zeros(np.shape(x0))
n = np.shape(Cpdf)[1] # Sample size

t0 = datetime.datetime.now()
t=1
MaxIter = 500
Epsilon = 1e-3
FractionalError = np.ones(MaxIter)
loglike = np.zeros(MaxIter)
print(t, np.sum(w), LogLikelihood(Cpdf, w, n))


while np.abs(FractionalError[t-1]) > Epsilon:
    if t==1:
        w = x0
    else:
        w = w_final

    TempMatrix =  Cpdf * w[:, None]
    IntMatrix = TempMatrix / np.sum(TempMatrix, axis=0)
    w_final = np.mean(IntMatrix, axis=1)
    '''
    for j in range(ReducedDeg**2):
        a = np.zeros(n)
        for i in range(n):
            a[i] = (Cpdf[j,i] * w[j])/np.matmul(Cpdf[:,i], w)
        w_final[j] = np.sum(a)/n
    '''
    loglike[t] = LogLikelihood(Cpdf, w_final, n)

    FractionalError[t] = (loglike[t] - loglike[t-1])/np.abs(loglike[t-1])
    print(t, np.sum(w_final), loglike[t], FractionalError[t])
    # print(loglike[0:10])
    t+=1

    if t == MaxIter:
        break

t1 = datetime.datetime.now()

print(t1-t0)

# np.matmul(w, Cpdf)

plt.figure()
plt.plot(loglike[np.nonzero(loglike)], '.')
plt.xlabel('Iteration')
plt.ylabel('LogLikelihood')
plt.tight_layout()

plt.figure()
plt.plot(FractionalError[np.nonzero(loglike)], '.')
plt.axhline(Epsilon, color='k', ls='--')
plt.xlabel('Iteration')
plt.ylabel('Absolute Fractional Error')
plt.tight_layout()

plt.figure()
plt.imshow(np.reshape(w_final - OldWeights, (ReducedDeg, ReducedDeg)), aspect='auto')
plt.colorbar()
plt.title("Difference in Weights\nOld LogLike = {:.2f}, New Log Like = {:.2f}".format(OldLogLike, loglike[np.nonzero(loglike)][-1]))
plt.tight_layout()

"""
