#%cd "C:/Users/shbhu/Documents/Git/Py_mass_radius_working/PyCode"

import numpy as np
from scipy.stats import beta,norm
from scipy.integrate import quad
from scipy.optimize import brentq as root
from astropy.table import Table
from scipy.optimize import minimize

t = Table.read('MR_Kepler_170605_noanalytTTV_noupplim.csv')
t = t.filled()

M_sigma = (abs(t['pl_masseerr1']) + abs(t['pl_masseerr2']))/2
R_sigma = (abs(t['pl_radeerr1']) + abs(t['pl_radeerr2']))/2

M_obs = np.array(t['pl_masse'])
R_obs = np.array(t['pl_rade'])

# bounds for Mass and Radius
Radius_min = -0.3
Radius_max = np.log10(max(R_obs) + np.std(R_obs)/np.sqrt(len(R_obs)))
Mass_min = np.log10(max(min(M_obs) - np.std(M_obs)/np.sqrt(len(M_obs)), 0.1))
Mass_max = np.log10(max(M_obs) + np.std(M_obs)/np.sqrt(len(M_obs)))
num_boot = 100
select_deg = 55

data = np.vstack((M_obs,R_obs)).T
sigma = np.vstack((M_sigma,R_sigma)).T

bounds = np.array([Mass_max,Mass_min,Radius_max,Radius_min])




x_max = 5
x_min = 0.01
x = 0.1
y = 0.17
y_max = 11
y_min = 0.03
w_hat = 1
deg = 5


def pdfnorm_beta(x, x_obs, x_sd, x_max, x_min, shape1, shape2, log = True):
    '''
    Product of normal and beta distribution
    '''
    if log == True:
        norm_beta = norm.pdf(x_obs, loc = 10**x, scale = x_sd) * beta.pdf((x - x_min)/(x_max - x_min), a = shape1, b = shape2)/(x_max - x_min)
    else:
        norm_beta = norm.pdf(x_obs, loc = x, scale = x_sd) * beta.pdf((x - x_min)/(x_max - x_min), a = shape1, b = shape2)/(x_max - x_min)     
        
    return norm_beta   
 
def integrate_function(data, data_sd, deg, degree, x_max, x_min, log = False, abs_tol = 1e-10):
    '''
    Integrate the product of the normal and beta distribution
    '''
    
    x_obs = data
    x_sd = data_sd
    shape1 = degree
    shape2 = deg - degree + 1
    log = log
    
    return quad(pdfnorm_beta, a = x_min, b = x_max,
                 args = (x_obs, x_sd, x_max, x_min, shape1, shape2, log), epsabs = abs_tol)[0]
       

def marginal_density(x, x_max, x_min, deg, w_hat):
    '''
    Calculate the marginal density 
    '''
    
    if type(x) == list:
        x = np.array(x)
    
    x_std = (x - x_min)/(x_max - x_min)
    deg_vec = np.arange(1,deg+1)    
    x_beta_indv = np.array([beta.pdf(x_std, a = d, b = deg - d + 1)/(x_max - x_min) for d in deg_vec])
    
    x_beta_pdf = np.kron(x_beta_indv, np.repeat(1,deg))
    
    marg_x = np.sum(w_hat * x_beta_pdf)
    
    return marg_x
    
    
def conditional_density(y, y_max, y_min, x, x_max, x_min, deg, w_hat):
    '''
    Calculate the conditional density
    '''
    if type(x) == list:
        x = np.array(x)
    if type(y) == list:
        y = np.array(y)
    
    deg_vec = np.arange(1,deg+1)  
    
    # Return Conditional Mean, Variance, Quantile, Distribution
    y_std = (y - y_min)/(y_max - y_min)
    x_std = (x - x_min)/(x_max - x_min)
    y_beta_indv = np.array([beta.pdf(y_std, a = d, b = deg - d + 1)/(y_max - y_min) for d in deg_vec])
    y_beta_pdf = np.kron(y_beta_indv, np.repeat(1,deg))
    
    denominator = np.sum(w_hat * y_beta_pdf)
    
    ########### Density ##########
    
    density_indv_pdf = np.array([beta.pdf(x_std, a = d, b = deg - d + 1)/(x_max - x_min) for d in deg_vec])
    density_pdf = w_hat * np.kron(density_indv_pdf,y_beta_indv)
    
    density = density_pdf / denominator
    
    return density
    
def cond_density_quantile(y, y_max, y_min, x_max, x_min, deg, w_hat, qtl = [0.16,0.84]):
    '''
    Calculate 16% and 84% quantiles of a conditional density
        
    '''    
    
    if type(y) == list:
        y = np.array(y)
    
    deg_vec = np.arange(1,deg+1)  
    
    y_std = (y - y_min)/(y_max - y_min)
    y_beta_indv = np.array([beta.pdf(y_std, a = d, b = deg - d + 1)/(y_max - y_min) for d in deg_vec])
    y_beta_pdf = np.kron(y_beta_indv, np.repeat(1,deg))
    
    denominator = np.sum(w_hat * y_beta_pdf)
    
    # Mean
    mean_beta_indv = (deg_vec * (x_max - x_min) / (deg + 1)) + x_min
    mean_beta = np.kron(mean_beta_indv,y_beta_indv)
    mean_nominator = np.sum(w_hat * mean_beta)
    mean = mean_nominator / denominator
    
    # Variance 
    var_beta_indv = (deg_vec * (deg - deg_vec + 1) * (x_max - x_min)**2 / ((deg + 2)*(deg + 1)**2)) 
    var_beta = np.kron(var_beta_indv,y_beta_indv)
    var_nominator = np.sum(w_hat * var_beta)
    var = var_nominator / denominator
    
    # Quantile
    
    def pbeta_conditional_density(x):
        
        def mix_density(j):
            
            x_indv_cdf = np.array([beta.cdf((j - x_min)/(x_max - x_min), a = d, b = deg - d + 1) for d in deg_vec])

            quantile_nominator = np.sum(w_hat * np.kron(x_indv_cdf,y_beta_indv))
            p_beta = quantile_nominator / denominator
            
            return p_beta

        if np.size(x)>1:
            return np.array([mix_density(i) for i in x])
        else:
            return mix_density(x)
            
            
    def conditional_quantile(q):
        def g(x):
            return pbeta_conditional_density(x) - q
        return root(g,a = x_min, b = x_max)
 
    quantile = [conditional_quantile(i) for i in qtl]
    
    return [mean,var,quantile]
    
    
######################################
##### Main function: MLE.fit #########
######################################
#' @param data: the first column contains the mass measurements and 
#'              the second column contains the radius measurements.
#' @param sigma: measurement errors for the data, if no measuremnet error, 
#'               it is NULL
#' @param bounds: a vector contains four elements, from left to right:
#'                the upper bound for mass, the lower bound for mass
#'                the upper bound for radius, the lower bound for radius
#' @param deg: degree used for the Bernstein polynomials
#' @param log: is the data transformed into a log scale
#' @param abs.tol: precision when calculate the integral 
#' @param output.weights.only only output the estimated weights from the
#'                            Bernstein polynomials if it is TRUE;
#'                            otherwise, output the conditional densities


def MLE_fit(data, bounds, deg, sigma = None, log = False,
                    abs_tol = 1e-20, output_weights_only = False):
    '''
    INPUT:
        data: The first column contains the mass measurements and 
              the second column contains the radius measurements.
              Numpy Array
        sigma: Measurement Errors for the Data. Default is None
        bounds: Vector with 4 elements. Upper and lower bound for Mass, Upper and lower bound for Radius.
        deg: Degree used for Bernstein polynomials
        log: If True, data is transformed into log scale
        abs_tol: Precision used to calculate integral
        output_weights_only: If True, only output the estimated weights from 
        the Bernstein polynomials. Else, output the conditional densities

    '''
    
    if np.shape(data)[0] < np.shape(data)[1]:
        data = np.transpose(data)
    
    # Read the Data
    
    n = np.shape(data)[0]
    M = data[:,0]
    R = data[:,1]
      
    if sigma is not None:
        sigma_M = sigma[:,0]
        sigma_R = sigma[:,1]
        
    M_max = bounds[0]
    M_min = bounds[1]
    R_max = bounds[2]
    R_min = bounds[3]
    
    deg_vec = np.arange(1,deg+1)     
    
    if sigma is None:
        # pdf for Mass and Radius for each beta density
        M_indv_pdf = np.array([beta.pdf((M - M_min)/(M_max - M_min), a = d, b = deg - d+1)/(M_max - M_min) for d in deg_vec])
        R_indv_pdf = np.array([beta.pdf((R - R_min)/(R_max - R_min), a = d, b = deg - d+1)/(R_max - R_min) for d in deg_vec])        
        
    else:
        print(deg)
        
        res_M_pdf = np.zeros((n,deg))
        res_R_pdf = np.zeros((n,deg))
        C_pdf = np.zeros((n,deg**2))
        
        for i in range(0,n):        
            for d in deg_vec:

                res_M_pdf[i,d-1] = integrate_function(data = M[i], data_sd = sigma_M[i], deg = deg, degree = d , x_max = M_max, x_min = M_min, log = log, abs_tol = abs_tol)
                res_R_pdf[i,d-1] = integrate_function(data = R[i], data_sd = sigma_R[i], deg = deg, degree = d , x_max = R_max, x_min = R_min, log = log, abs_tol = abs_tol)

            # put M.indv.pdf and R.indv.pdf into a big matrix
            C_pdf[i,:] = np.kron(res_M_pdf[i],res_R_pdf[i])

            
        C_pdf = C_pdf.T 
       
    # Function input to optimizer            
    def fn1(w):
        return - np.sum(np.log(np.matmul(w,C_pdf)))
  
        
    def eqn(w):
        return np.sum(w) - 1
    
    def eqn_jacobian(w):
        return len(w)
        
    #        (fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)
        
    eq_cons = {'type': 'eq',
               'fun' : eqn,
               'jac' : eqn_jacobian} 

    bounds = [[0,1]]*deg**2
     
    res = minimize(fun = fn1, x0 = np.repeat(1./(deg**2),deg**2) , jac = False, method='SLSQP',
                   constraints = [eq_cons], bounds =  bounds, options={'ftol': 1e-9, 'disp': True} )
    
    '''    
      opt.w <- solnp(rep(1/(deg^2), deg^2), 
                 fun = fn1, eqfun = eqn, eqB = 1, 
                 LB = rep(0, deg^2), UB = rep(1, deg^2), 
                 control=list(trace = 0))
        
    '''

        
    

