import numpy as np
from scipy.stats import beta,norm
import scipy
from scipy.integrate import quad
from scipy.optimize import brentq as root
import datetime,os
from multiprocessing import current_process


from mrexo.utils import _logging
from mrexo.Optimizers import optimizer


########################################
##### Main function: MLE_fit() #########
########################################

def MLE_fit(X, X_sigma, Y, Y_sigma,
            X_bounds, Y_bounds, Y_char, X_char,
            deg, Log=True, abs_tol=1e-8, output_weights_only=False,
            save_path=None, calc_joint_dist = False, verbose=2):
    '''
    Perform maximum likelihood estimation to find the weights for the beta density basis functions.
    Also, use those weights to calculate the conditional density distributions.
    Ning et al. 2018 Sec 2.2, Eq 9.

    INPUT:
        Y: Numpy array of Y measurements. In LINEAR SCALE.
        Y_sigma: Numpy array of Y uncertainties.
            Assumes symmetrical uncertainty. In LINEAR SCALE.
        X: Numpy array of X measurements. In LINEAR SCALE.
        X_sigma: Numpy array of X uncertainties.
            Assumes symmetrical uncertainty. In LINEAR SCALE.
        Y_bounds: Bounds for the Y. Log10
        X_bounds: Bounds for the X. Log10
        X_char: String alphabet (character) to depict X quantity.
            Eg 'm' for Mass, 'r' for Radius
        Y_char: String alphabet (character) to depict Y quantity
            Eg 'm' for Mass, 'r' for Radius
        deg: Degree used for beta densities polynomials. Integer value.
        Log: If True, data is transformed into Log scale. Default=True,
            since the fitting function always converts data to log scale.
        abs_tol: Absolute tolerance to be used for the numerical integration
            for product of normal and beta distribution. Default : 1e-8
        output_weights_only: If True, only output the estimated weights,
            else will also output dictionary with keys shown below.
        save_path: Location of folder for auxiliary output files.
        calc_joint_dist: If True, will calculate and output the
            joint distribution of Y and X.
        verbose: Integer specifying verbosity for logging.
                If 0: Will not log in the log file or print statements.
                If 1: Will write log file only.
                If 2: Will write log file and print statements.

    \nOUTPUT:

        If output_weights_only == True,
        w_hat : Weights for the beta densities.

        If output_weights_only == False,
        output: Output dictionary from fitting using
                Maximum Likelihood Estimation.
                The keys in the dictionary are:
                'weights' : Weights for beta densities.
                'aic' : Akaike Information Criterion.
                'bic' : Bayesian Information Criterion.
                'Y_points' : Sequence of Y points for
                    initial fitting w/o bootstrap.
                'X_points' : Sequence of X points for
                    initial fitting w/o bootstrap.
                'Y_cond_X' : Conditional distribution of Y given X.
                'Y_cond_X_var' : Variance for the Conditional distribution
                    of Y given X.
                'Y_cond_X_quantile' : Quantiles for the Conditional distribution
                    of Y given X.
                'X_cond_Y' : Conditional distribution of X given Y.
                'X_cond_Y_var' : Variance for the Conditional distribution
                    of X given Y.
                'X_cond_Y_quantile' : Quantiles for the Conditional distribution
                    of X given Y.


                if calc_joint_dist == True:
                'joint_dist' : Joint distribution of Y AND X.
    EXAMPLE:

            result = MLE_fit(y=Y, x=X, Y_sigma=Y_sigma,
                            X_sigma=X_sigma,
                            y_bounds=Y_bounds, x_bounds=X_bounds,
                            deg=int(deg_choose), abs_tol=abs_tol,
                            save_path=aux_output_location)
    '''
    starttime = datetime.datetime.now()
    if save_path is None:
        save_path = os.path.dirname(__file__)


    message = '====\nStarted run at {}\n'.format(starttime)
    _ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)


    n = np.shape(Y)[0]
    Y_max = Y_bounds[1]
    Y_min = Y_bounds[0]
    X_max = X_bounds[1]
    X_min = X_bounds[0]

    ########################################################################
    # Integration to find C matrix (input for log likelihood maximization.)
    ########################################################################
    C_pdf = calc_C_matrix(n=n, deg=deg, Y=Y, Y_sigma=Y_sigma, Y_max=Y_max, Y_min=Y_min,
                        X=X, X_sigma=X_sigma, X_max=X_max, X_min=X_min,
                        Log=Log, abs_tol=abs_tol, save_path=save_path, verbose=verbose, SaveCMatrix=False)

    message = 'Finished Integration at {}. \nCalculated the PDF for {} and {} for Integrated beta and normal density.\n'.format(datetime.datetime.now(), Y_char, X_char)
    _ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)


    ###########################################################
    # Run optimization to find the weights
    ###########################################################


    unpadded_weight, n_log_lik = optimizer(C_pdf=C_pdf, deg=deg,
                    verbose=verbose, save_path=save_path)
    print("AAAAAAAAAAAAAAAA   {}".format(n_log_lik))
    # rand = np.random.randn()
    # np.savetxt(os.path.join(save_path, 'loglikelihoodtest{:.3f}.txt'.format(rand)), [n_log_lik])
    # np.savetxt(os.path.join(save_path, 'Cpdf{:.3f}.txt'.format(rand)), C_pdf)
    # np.savetxt(os.path.join(save_path, 'IntermediateWeight{:.3f}.txt'.format(rand)), w_hat)

    # Pad the weight array with zeros for the
    w_sq = np.reshape(unpadded_weight,[deg-2,deg-2])
    w_sq_padded = np.zeros((deg,deg))
    w_sq_padded[1:-1,1:-1] = w_sq
    w_hat = w_sq_padded.flatten()


    if output_weights_only == True:
        return unpadded_weight

    else:
        # Calculate AIC and BIC
        aic = n_log_lik*2 + 2*(deg**2 - 1)
        bic = n_log_lik*2 + np.log(n)*(deg**2 - 1)

        Y_seq = np.linspace(Y_min,Y_max,100)
        X_seq = np.linspace(X_min,X_max,100)

        output = {'weights': w_hat,
                  'aic': aic,
                  'bic': bic,
                  'Y_points': Y_seq,
                  'X_points': X_seq}


        deg_vec = np.arange(1,deg+1)

        Y_cond_X_median, Y_cond_X_var, Y_cond_X_quantile = [], [], []
        X_cond_Y_median, X_cond_Y_var, X_cond_Y_quantile = [], [], []

        for i in range(0,len(X_seq)):
            # Conditional Densities with 16% and 84% quantile
            Y_cond_X = cond_density_quantile(a = X_seq[i], a_max = X_max, a_min = X_min,
                            b_max = Y_max, b_min = Y_min, deg = deg, deg_vec = deg_vec, w_hat = w_hat, qtl = [0.5,0.16,0.84])[0:3]
            Y_cond_X_median.append(Y_cond_X[2][0])
            Y_cond_X_var.append(Y_cond_X[1])
            Y_cond_X_quantile.append(Y_cond_X[2][1:])

            X_cond_Y = cond_density_quantile(a = Y_seq[i], a_max=Y_max, a_min=Y_min,
                                b_max=X_max, b_min=X_min, deg=deg, deg_vec = deg_vec,
                                w_hat=np.reshape(w_hat,(deg,deg)).T.flatten(), qtl = [0.5,0.16,0.84])[0:3]
            X_cond_Y_median.append(X_cond_Y[2][0])
            X_cond_Y_var.append(X_cond_Y[1])
            X_cond_Y_quantile.append(X_cond_Y[2][1:])



        # Output everything as dictionary

        output['Y_cond_X'] = Y_cond_X_median
        output['Y_cond_X_var'] = Y_cond_X_var
        output['Y_cond_X_quantile'] = np.array(Y_cond_X_quantile)
        output['X_cond_Y'] = X_cond_Y_median
        output['X_cond_Y_var'] = X_cond_Y_var
        output['X_cond_Y_quantile'] = np.array(X_cond_Y_quantile)

        if calc_joint_dist == True:
            joint_dist = calculate_joint_distribution(X_seq, X_min, X_max, Y_seq, Y_min, Y_max, w_hat, abs_tol)
            output['joint_dist'] = joint_dist

        return output


def calc_C_matrix(n, deg, Y, Y_sigma, Y_max, Y_min, X, X_sigma, X_max, X_min, abs_tol, save_path, Log, verbose, SaveCMatrix=False):
    '''
    Integrate the product of the normal and beta distributions for Y and X and then take the Kronecker product.

    Refer to Ning et al. 2018 Sec 2.2 Eq 8 and 9.

    \nINPUTS:
        n: Number of data points
        deg: Degree used for beta densities
        Y: Numpy array of y measurements. In LINEAR SCALE.
        Y_sigma: Numpy array of y uncertainties. Assumes symmetrical uncertainty. In LINEAR SCALE.
        Y_max, Y_min : Maximum and minimum value for y. Log10
        X: Numpy array of X measurements. In LINEAR SCALE.
        X_sigma: Numpy array of x uncertainties. Assumes symmetrical uncertainty. In LINEAR SCALE.
        X_max, X_min : Maximum and minimum value for x. Log10
        abs_tol: Absolute tolerance to be used for the numerical integration for product of normal and beta distribution.
                Default : 1e-8
        save_path: Location of folder within results for auxiliary output files
        Log: If True, data is transformed into Log scale. Default=True, since
            fitting function always converts data to log scale.
        verbose: Integer specifying verbosity for logging.
            If 0: Will not log in the log file or print statements.
            If 1: Will write log file only.
            If 2: Will write log file and print statements.

    OUTPUTS:

        C_pdf : Matrix explained in Ning et al. Equation 8. Product of (integrals of (product of normal and beta
                distributions)) for  Y and x.
    '''
    deg_vec = np.arange(2,deg)

    Y_indv_pdf = np.zeros((n, deg-2))
    X_indv_pdf = np.zeros((n, deg-2))
    C_pdf = np.zeros((n, (deg-2)**2))


    message = 'Started Integration at {}\n'.format(datetime.datetime.now())
    _ = _logging(message=message, filepath=save_path, verbose=verbose, append=True)



    # Loop across each data point.
    for i in range(0,n):
        Y_indv_pdf[i,:] = _find_indv_pdf(Y[i], deg, deg_vec, Y_max, Y_min, Y_sigma[i], abs_tol=abs_tol, Log=Log)
        X_indv_pdf[i,:] = _find_indv_pdf(X[i], deg, deg_vec, X_max, X_min, X_sigma[i], abs_tol=abs_tol, Log=Log)
        # print(M[i],Y_sigma[i], R[i], X_sigma[i], Y_max, Y_min, X_max, X_min, np.sum(R_indv_pdf[i,:]))

        # Put M.indv.pdf and R.indv.pdf into a big matrix
        C_pdf[i,:] = np.kron(Y_indv_pdf[i], X_indv_pdf[i])

    C_pdf = C_pdf.T

    # Log of 0 throws weird errors
    C_pdf[C_pdf == 0] = 1e-300
    C_pdf[np.where(np.isnan(C_pdf))] = 1e-300

    if SaveCMatrix:
        np.savetxt(os.path.join(save_path, 'C_pdf.txt'), C_pdf)
    return C_pdf


def _norm_pdf(a, loc, scale):
    '''
    Find the PDF for a normal distribution. Identical to scipy.stats.norm.pdf.
    Runs much quicker without the generic function handling.
    CHECK'''
    N = (a - loc)/scale
    return np.exp(-N*N/2)/(np.sqrt(2*np.pi))/scale

def _int_gamma(a):
    return scipy.math.factorial(a-1)


def _beta_pdf(x,a,b):
    f = (_int_gamma(a+b) * x**(a-1)*(1-x)**(b-1))/(_int_gamma(a)*_int_gamma(b))
    return f


def _pdfnorm_beta(a, a_obs, a_std, a_max, a_min, shape1, shape2, Log=True):
    '''
    Product of normal and beta distribution

    Refer to Ning et al. 2018 Sec 2.2, Eq 8.
    CHECK'''

    if Log == True:
        norm_beta = _norm_pdf(a_obs, loc=10**a, scale=a_std) * _beta_pdf((a - a_min)/(a_max - a_min), a=shape1, b=shape2)/(a_max - a_min)
    else:
        norm_beta = _norm_pdf(a_obs, loc=a, scale=a_std) * _beta_pdf((a - a_min)/(a_max - a_min), a=shape1, b=shape2)/(a_max - a_min)
    return norm_beta

def integrate_function(data, data_std, deg, degree, a_max, a_min, Log=False, abs_tol=1e-8):
    '''
    Integrate the product of the normal and beta distribution.

    Refer to Ning et al. 2018 Sec 2.2, Eq 8.
    CHECK'''
    a_obs = data
    a_std = data_std
    shape1 = degree
    shape2 = deg - degree + 1
    Log = Log

    integration_product = quad(_pdfnorm_beta, a=a_min, b=a_max,
                          args=(a_obs, a_std, a_max, a_min, shape1, shape2, Log), epsabs = abs_tol, epsrel = 1e-8)
    return integration_product[0]


def _find_indv_pdf(a, deg, deg_vec, a_max, a_min, a_std=np.nan, abs_tol=1e-8, Log=True):
    '''
    Find the individual probability density Function for a variable.
    If the data has uncertainty, the joint distribution is modelled using a
    convolution of beta and normal distributions.

    Refer to Ning et al. 2018 Sec 2.2, Eq 8.
    CHECK'''


    if np.isnan(a_std):
        if Log:
            a_std = (np.log10(a) - a_min)/(a_max - a_min)
        else:
            a_std = (a - a_min)/(a_max - a_min)
        a_beta_indv = np.array([_beta_pdf(a_std, a=d, b=deg - d + 1)/(a_max - a_min) for d in deg_vec])
    else:
        a_beta_indv = np.array([integrate_function(data=a, data_std=a_std, deg=deg, degree=d, a_max=a_max, a_min=a_min, abs_tol=abs_tol, Log=Log) for d in deg_vec])
    return a_beta_indv


def _marginal_density(a, a_max, a_min, deg, w_hat):
    '''
    Calculate the marginal density

    Refer to Ning et al. 2018 Sec 2.2, Eq 10
    '''
    if type(a) == list:
        a = np.array(a)

    deg_vec = np.arange(1,deg+1)
    x_beta_indv = _find_indv_pdf(a,deg, deg_vec, a_max, a_min)
    x__beta_pdf = np.kron(x_beta_indv, np.repeat(1,deg))

    marg_x = np.sum(w_hat * x__beta_pdf)

    return marg_x

def cond_density_quantile(a, a_max, a_min, b_max, b_min, deg, deg_vec, w_hat, a_std=np.nan, qtl=[0.16,0.84], abs_tol=1e-8):
    '''
    Calculate 16% and 84% quantiles of a conditional density, along with the mean and variance.

    Refer to Ning et al. 2018 Sec 2.2, Eq 10
    '''
    if type(a) == list:
        a = np.array(a)

    a_beta_indv = _find_indv_pdf(a=a, deg=deg, deg_vec=deg_vec, a_max=a_max, a_min=a_min, a_std=a_std, abs_tol=abs_tol, Log=False)
    a_beta_pdf = np.kron(np.repeat(1,np.max(deg_vec)),a_beta_indv)

    # Equation 10b Ning et al 2018
    denominator = np.sum(w_hat * a_beta_pdf)

    if denominator == 0:
        denominator = np.nan

    # Mean
    mean_beta_indv = (deg_vec * (b_max - b_min) / (deg + 1)) + b_min
    mean_beta = np.kron(mean_beta_indv,a_beta_indv)
    mean_numerator = np.sum(w_hat * mean_beta)
    mean = mean_numerator / denominator

    # Variance
    var_beta_indv = (deg_vec * (deg - deg_vec + 1) * (b_max - b_min)**2 / ((deg + 2)*(deg + 1)**2))
    var_beta = np.kron(var_beta_indv,a_beta_indv)
    var_numerator = np.sum(w_hat * var_beta)
    var = var_numerator / denominator

    # Quantile

    def pbeta_conditional_density(j):
        if type(j) == np.ndarray:
            j = j[0]
        b_indv_cdf = np.array([beta.cdf((j - b_min)/(b_max - b_min), a=d, b=deg - d + 1) for d in deg_vec])
        quantile_numerator = np.sum(w_hat * np.kron(b_indv_cdf,a_beta_indv))
        p_beta = quantile_numerator / denominator

        return p_beta


    def conditional_quantile(q):
        def g(x):
            return pbeta_conditional_density(x) - q
        return root(g, a=b_min, b=b_max, xtol=1e-8, rtol=1e-12)


    if np.size(qtl) == 1:
        qtl = [qtl]
    quantile = [conditional_quantile(i) for i in qtl]

    return mean, var, quantile, denominator, a_beta_indv


def calculate_joint_distribution(X_points, X_min, X_max, Y_points, Y_min, Y_max, weights, abs_tol):
    '''
    Calculcate the joint distribution of Y and X (Y and X) : f(y,x|w,d,d')
    Refer to Ning et al. 2018 Sec 2.1, Eq 7
    '''

    deg = int(np.sqrt(len(weights)))
    deg_vec = np.arange(1,deg+1)

    joint = np.zeros((len(X_points), len(Y_points)))

    for i in range(len(X_points)):
        for j in range(len(Y_points)):
                    x_beta_indv = _find_indv_pdf(a=X_points[i], deg=deg, deg_vec=deg_vec, a_max=X_max, a_min=X_min, a_std=np.nan, abs_tol=abs_tol, Log=False)
                    y_beta_indv = _find_indv_pdf(a=Y_points[j], deg=deg, deg_vec=deg_vec, a_max=Y_max, a_min=Y_min, a_std=np.nan, abs_tol=abs_tol, Log=False)

                    intermediate = np.matmul(np.reshape(weights,(deg,deg)),np.matrix(x_beta_indv).T)
                    joint[i,j] = np.matmul(np.matrix(y_beta_indv), intermediate)

    return joint.T
