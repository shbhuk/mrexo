import numpy as np
from scipy.stats import beta,norm
import scipy
from scipy.integrate import quad
from scipy.optimize import brentq as root
from scipy.optimize import fmin_slsqp
import datetime,os

########################################
##### Main function: MLE_fit() #########
########################################

def MLE_fit(Mass, Mass_sigma, Radius, Radius_sigma, Mass_bounds, Radius_bounds,
            deg, Log=True, abs_tol=1e-8, output_weights_only=False,
            save_path=None, calc_joint_dist = False):
    '''
    Perform maximum likelihood estimation to find the weights for the beta density basis functions.
    Also, use those weights to calculate the conditional density distributions.
    Ning et al. 2018 Sec 2.2, Eq 9.

    INPUT:
        Mass: Numpy array of mass measurements. In LINEAR SCALE.
        Mass_sigma: Numpy array of mass uncertainties. Assumes symmetrical uncertainty. In LINEAR SCALE.
        Radius: Numpy array of radius measurements. In LINEAR SCALE.
        Radius_sigma: Numpy array of radius uncertainties. Assumes symmetrical uncertainty. In LINEAR SCALE.
        Mass_bounds: Bounds for the mass. Log10
        Radius_bounds: Bounds for the radius. Log10
        deg: Degree used for beta densities polynomials. Integer value.
        Log: If True, data is transformed into Log scale. Default=True, since the
            fitting function always converts data to log scale.
        abs_tol : Absolute tolerance to be used for the numerical integration for product of normal and beta distribution.
                Default : 1e-8
        output_weights_only: If True, only output the estimated weights, else will also output dictionary with keys shown below.
        save_path: Location of folder within results for auxiliary output files.
        calc_joint_dist: If True, will calculate and output the joint distribution of mass and radius.

    OUTPUT:
        If output_weights_only == True,
        w_hat : Weights for the beta densities.

        If output_weights_only == False,
        output: Output dictionary from fitting using Maximum Likelihood Estimation.
                The keys in the dictionary are:
                'weights' : Weights for beta densities.
                'aic' : Akaike Information Criterion.
                'bic' : Bayesian Information Criterion.
                'M_points' : Sequence of mass points for initial fitting w/o bootstrap.
                'R_points' : Sequence of radius points for initial fitting w/o bootstrap.
                'M_cond_R' : Conditional distribution of mass given radius.
                'M_cond_R_var' : Variance for the Conditional distribution of mass given radius.
                'M_cond_R_quantile' : Quantiles for the Conditional distribution of mass given radius.
                'R_cond_M' : Conditional distribution of radius given mass.
                'R_cond_M_var' : Variance for the Conditional distribution of radius given mass.
                'R_cond_M_quantile' : Quantiles for the Conditional distribution of radius given mass.


                if calc_joint_dist == True:
                'joint_dist' : Joint distribution of mass AND radius.
    EXAMPLE:
            result = MLE_fit(Mass=Mass, Radius=Radius, Mass_sigma=Mass_sigma, Radius_sigma=Radius_sigma,
                            Mass_bounds=Mass_bounds, Radius_bounds=Radius_bounds,  deg=int(deg_choose), abs_tol=abs_tol, save_path=aux_output_location)
    '''
    print('New MLE')
    starttime = datetime.datetime.now()
    if save_path is None:
        save_path = os.path.dirname(__file__)
    with open(os.path.join(save_path,'log_file.txt'),'a') as f:
       f.write('\n======================================\n')
       f.write('Started run at {}\n'.format(starttime))

    n = np.shape(Mass)[0]
    Mass_max = Mass_bounds[1]
    Mass_min = Mass_bounds[0]
    Radius_max = Radius_bounds[1]
    Radius_min = Radius_bounds[0]

    ###########################################################
    # Integration to find C matrix (input for log likelihood maximization.)
    ###########################################################
    C_pdf = calc_C_matrix(n=n, deg=deg, M=Mass, Mass_sigma=Mass_sigma, Mass_max=Mass_max, Mass_min=Mass_min,
                        R=Radius, Radius_sigma=Radius_sigma, Radius_max=Radius_max, Radius_min=Radius_min,
                        Log=Log, abs_tol=abs_tol, save_path=save_path)

    print('Finished Integration at ',datetime.datetime.now())
    with open(os.path.join(save_path,'log_file.txt'),'a') as f:
        f.write('Finished Integration at {}\n'.format(datetime.datetime.now()))

    print('Calculated the PDF for Mass and Radius for Integrated beta and normal density')

    ###########################################################
    # Run optimization to find the weights
    ###########################################################

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
    x0 = np.repeat(1./(deg**2),(deg-2)**2)

    # Run optimization to find optimum value for each degree (weights). These are the coefficients for the beta densities being used as a linear basis.
    opt_result = fmin_slsqp(fn1, x0, bounds=bounds, f_eqcons=eqn, iter=250, full_output=True, iprint=1,
                            epsilon=1e-5, acc=1e-5)
    print('Optimization run finished at {}, with {} iterations. Exit Code = {}\n\n'.format(datetime.datetime.now(),
            opt_result[2], opt_result[3], opt_result[4]))





    with open(os.path.join(save_path,'log_file.txt'),'a') as f:
        f.write('Finished Optimization at {}'.format(datetime.datetime.now()))
        f.write('\nOptimization terminated after {} iterations. Exit Code = {}{}\n\n'.format(opt_result[2],opt_result[3],opt_result[4]))

    unpadded_weight = opt_result[0]
    n_log_lik = opt_result[1]

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

        M_seq = np.linspace(Mass_min,Mass_max,100)
        R_seq = np.linspace(Radius_min,Radius_max,100)

        output = {'weights': w_hat,
                  'aic': aic,
                  'bic': bic,
                  'M_points': M_seq,
                  'R_points': R_seq}


        deg_vec = np.arange(1,deg+1)

        M_cond_R_median, M_cond_R_var, M_cond_R_quantile = [], [], []
        R_cond_M_median, R_cond_M_var, R_cond_M_quantile = [], [], []

        for i in range(0,len(R_seq)):
            # Conditional Densities with 16% and 84% quantile
            M_cond_R = cond_density_quantile(y = R_seq[i], y_max = Radius_max, y_min = Radius_min,
                            x_max = Mass_max, x_min = Mass_min, deg = deg, deg_vec = deg_vec, w_hat = w_hat, qtl = [0.5,0.16,0.84])[0:3]
            M_cond_R_median.append(M_cond_R[2][0])
            M_cond_R_var.append(M_cond_R[1])
            M_cond_R_quantile.append(M_cond_R[2][1:])

            R_cond_M = cond_density_quantile(y = M_seq[i], y_max=Mass_max, y_min=Mass_min,
                                x_max=Radius_max, x_min=Radius_min, deg=deg, deg_vec = deg_vec,
                                w_hat=np.reshape(w_hat,(deg,deg)).T.flatten(), qtl = [0.5,0.16,0.84])[0:3]
            R_cond_M_median.append(R_cond_M[2][0])
            R_cond_M_var.append(R_cond_M[1])
            R_cond_M_quantile.append(R_cond_M[2][1:])



        # Output everything as dictionary

        output['M_cond_R'] = M_cond_R_median
        output['M_cond_R_var'] = M_cond_R_var
        output['M_cond_R_quantile'] = np.array(M_cond_R_quantile)
        output['R_cond_M'] = R_cond_M_median
        output['R_cond_M_var'] = R_cond_M_var
        output['R_cond_M_quantile'] = np.array(R_cond_M_quantile)

        if calc_joint_dist == True:
            joint_dist = calculate_joint_distribution(R_seq, Radius_min, Radius_max, M_seq, Mass_min, Mass_max, w_hat, abs_tol)
            output['joint_dist'] = joint_dist

        return output


def calc_C_matrix(n, deg, M, Mass_sigma, Mass_max, Mass_min, R, Radius_sigma, Radius_max, Radius_min, abs_tol, save_path, Log):
    '''
    Integrate the product of the normal and beta distributions for mass and radius and then take the Kronecker product.

    Refer to Ning et al. 2018 Sec 2.2 Eq 8 and 9.

    INPUTS:
        n: Number of data points
        deg: Degree used for beta densities
        Mass: Numpy array of mass measurements. In LINEAR SCALE.
        Mass_sigma: Numpy array of mass uncertainties. Assumes symmetrical uncertainty. In LINEAR SCALE.
        Mass_max, Mass_min : Maximum and minimum value for mass. Log10
        Radius: Numpy array of radius measurements. In LINEAR SCALE.
        Radius_sigma: Numpy array of radius uncertainties. Assumes symmetrical uncertainty. In LINEAR SCALE.
        Radius_max, Radius_min : Maximum and minimum value for radius. Log10
        abs_tol : Absolute tolerance to be used for the numerical integration for product of normal and beta distribution.
                Default : 1e-8
        save_path: Location of folder within results for auxiliary output files
        Log: If True, data is transformed into Log scale. Default=True, since
            fitting function always converts data to log scale.

    OUTPUTS:
        C_pdf : Matrix explained in Ning et al. Equation 8. Product of (integrals of (product of normal and beta
                distributions)) for mass and radius.
    '''
    deg_vec = np.arange(2,deg)

    M_indv_pdf = np.zeros((n, deg-2))
    R_indv_pdf = np.zeros((n, deg-2))
    C_pdf = np.zeros((n, (deg-2)**2))

    print('Started Integration at ',datetime.datetime.now())
    with open(os.path.join(save_path,'log_file.txt'),'a') as f:
        f.write('Started Integration at {}\n'.format(datetime.datetime.now()))

    # Loop across each data point.
    for i in range(0,n):
        M_indv_pdf[i,:] = find_indv_pdf(M[i], deg, deg_vec, Mass_max, Mass_min, Mass_sigma[i], abs_tol=abs_tol, Log=Log)
        R_indv_pdf[i,:] = find_indv_pdf(R[i], deg, deg_vec, Radius_max, Radius_min, Radius_sigma[i], abs_tol=abs_tol, Log=Log)

        # Put M.indv.pdf and R.indv.pdf into a big matrix
        C_pdf[i,:] = np.kron(M_indv_pdf[i], R_indv_pdf[i])

    C_pdf = C_pdf.T

    # Log of 0 throws weird errors
    C_pdf[C_pdf == 0] = 1e-300
    C_pdf[np.where(np.isnan(C_pdf))] = 1e-300
    return C_pdf


def norm_pdf(x, loc, scale):
    '''
    Find the PDF for a normal distribution. Identical to scipy.stats.norm.pdf.
    Runs much quicker without the generic function handling.
    '''
    y = (x - loc)/scale
    return np.exp(-y*y/2)/(np.sqrt(2*np.pi))/scale

def int_gamma(a):
    return scipy.math.factorial(a-1)


def beta_pdf(x,a,b):
    f = (int_gamma(a+b) * x**(a-1)*(1-x)**(b-1))/(int_gamma(a)*int_gamma(b))
    return f


def pdfnorm_beta(x, x_obs, x_std, x_max, x_min, shape1, shape2, Log=True):
    '''
    Product of normal and beta distribution

    Refer to Ning et al. 2018 Sec 2.2, Eq 8.
    '''

    if Log == True:
        norm_beta = norm_pdf(x_obs, loc=10**x, scale=x_std) * beta_pdf((x - x_min)/(x_max - x_min), a=shape1, b=shape2)/(x_max - x_min)
    else:
        norm_beta = norm_pdf(x_obs, loc=x, scale=x_std) * beta_pdf((x - x_min)/(x_max - x_min), a=shape1, b=shape2)/(x_max - x_min)

    return norm_beta

def integrate_function(data, data_std, deg, degree, x_max, x_min, Log=False, abs_tol=1e-8):
    '''
    Integrate the product of the normal and beta distribution.

    Refer to Ning et al. 2018 Sec 2.2, Eq 8.
    '''
    x_obs = data
    x_std = data_std
    shape1 = degree
    shape2 = deg - degree + 1
    Log = Log

    integration_product = quad(pdfnorm_beta, a=x_min, b=x_max,
                          args=(x_obs, x_std, x_max, x_min, shape1, shape2, Log), epsabs = abs_tol, epsrel = 1e-8)

    return integration_product[0]


def find_indv_pdf(x,deg,deg_vec,x_max,x_min,x_std=None, abs_tol=1e-8, Log=True):
    '''
    Find the individual probability density Function for a variable.
    When the data has uncertainty, the joint distribution is modelled using a
    convolution of beta and normal distributions.

    Refer to Ning et al. 2018 Sec 2.2, Eq 7 & 8.
    '''

    if x_std == None:
        x_std = (x - x_min)/(x_max - x_min)
        x_beta_indv = np.array([beta_pdf(x_std, a=d, b=deg - d + 1)/(x_max - x_min) for d in deg_vec])
    else:
        x_beta_indv = np.array([integrate_function(data=x, data_std=x_std, deg=deg, degree=d, x_max=x_max, x_min=x_min, abs_tol=abs_tol, Log=Log) for d in deg_vec])

    return x_beta_indv


def marginal_density(x, x_max, x_min, deg, w_hat):
    '''
    Calculate the marginal density

    Refer to Ning et al. 2018 Sec 2.2, Eq 10
    '''
    if type(x) == list:
        x = np.array(x)

    deg_vec = np.arange(1,deg+1)
    x_beta_indv = find_indv_pdf(x,deg, deg_vec, x_max, x_min)
    x_beta_pdf = np.kron(x_beta_indv, np.repeat(1,deg))

    marg_x = np.sum(w_hat * x_beta_pdf)

    return marg_x

def cond_density_quantile(y, y_max, y_min, x_max, x_min, deg, deg_vec, w_hat, y_std=None, qtl=[0.16,0.84], abs_tol=1e-8):
    '''
    Calculate 16% and 84% quantiles of a conditional density, along with the mean and variance.

    Refer to Ning et al. 2018 Sec 2.2, Eq 10
    '''
    if type(y) == list:
        y = np.array(y)

    y_beta_indv = find_indv_pdf(x=y, deg=deg, deg_vec=deg_vec, x_max=y_max, x_min=y_min, x_std=y_std, abs_tol=abs_tol, Log=False)
    y_beta_pdf = np.kron(np.repeat(1,np.max(deg_vec)),y_beta_indv)

    # Equation 10b Ning et al 2018
    denominator = np.sum(w_hat * y_beta_pdf)

    if denominator == 0:
        denominator = np.nan

    # Mean
    mean_beta_indv = (deg_vec * (x_max - x_min) / (deg + 1)) + x_min
    mean_beta = np.kron(mean_beta_indv,y_beta_indv)
    mean_numerator = np.sum(w_hat * mean_beta)
    mean = mean_numerator / denominator

    # Variance
    var_beta_indv = (deg_vec * (deg - deg_vec + 1) * (x_max - x_min)**2 / ((deg + 2)*(deg + 1)**2))
    var_beta = np.kron(var_beta_indv,y_beta_indv)
    var_numerator = np.sum(w_hat * var_beta)
    var = var_numerator / denominator

    # Quantile

    def pbeta_conditional_density(j):
        x_indv_cdf = np.array([beta.cdf((j - x_min)/(x_max - x_min), a=d, b=deg - d + 1) for d in deg_vec])

        quantile_numerator = np.sum(w_hat * np.kron(x_indv_cdf,y_beta_indv))
        p_beta = quantile_numerator / denominator

        return p_beta


    def conditional_quantile(q):
        def g(x):
            return pbeta_conditional_density(x) - q
        return root(g,a=x_min, b=x_max, xtol=1e-8, rtol=1e-12)

    if np.size(qtl) == 1:
        qtl = [qtl]
    quantile = [conditional_quantile(i) for i in qtl]

    return mean, var, quantile, denominator, y_beta_indv


def calculate_joint_distribution(R_points, Radius_min, Radius_max, M_points, Mass_min, Mass_max, weights, abs_tol):
    '''
    Calculcate the joint distribution of mass and radius : f(m,r|w,d,d')
    Refer to Ning et al. 2018 Sec 2.1, Eq 7
    '''

    deg = int(np.sqrt(len(weights)))
    deg_vec = np.arange(1,deg+1)

    joint = np.zeros((len(R_points), len(M_points)))

    for i in range(len(R_points)):
        for j in range(len(M_points)):
                    r_beta_indv = find_indv_pdf(x=R_points[i], deg=deg, deg_vec=deg_vec, x_max=Radius_max, x_min=Radius_min, x_std=None, abs_tol=abs_tol, Log=False)
                    m_beta_indv = find_indv_pdf(x=M_points[j], deg=deg, deg_vec=deg_vec, x_max=Mass_max, x_min=Mass_min, x_std=None, abs_tol=abs_tol, Log=False)

                    intermediate = np.matmul(np.reshape(weights,(deg,deg)),np.matrix(r_beta_indv).T)
                    joint[i,j] = np.matmul(np.matrix(m_beta_indv), intermediate)

    return joint.T
