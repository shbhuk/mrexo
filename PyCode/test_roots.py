
import numpy as np
from scipy.stats import beta,norm
from scipy.integrate import quad
from scipy.optimize import brentq as root

    
   

w_hat = np.loadtxt('Test/what.txt')
R_max , R_min = np.loadtxt('Test/R_bounds.txt')
M_max , M_min = np.loadtxt('Test/M_bounds.txt')
M_seq = np.loadtxt('Test/M_seq.txt')
R_seq = np.loadtxt('Test/R_seq.txt')
deg = 5


x_max = M_max
x_min = M_min
y_max = R_max
y_min = R_min


i = 1
y = 0.17
x = 0.1


"""
def cond_density_quantile(y, y_max, y_min, x_max, x_min, deg, w_hat, qtl = [0.16,0.84]):
    '''
    Calculate 16% and 84% quantiles of a conditional density
        
    '''  
"""  
qtl = [0.16,0.84]
if type(y) == list:
    y = np.array(y)

deg_vec = np.arange(1,deg+1)  

y_std = (y - y_min)/(y_max - y_min)
y_beta_indv = np.array([beta.pdf(y_std, a = d, b = deg - d + 1)/(y_max - y_min) for d in deg_vec])
y_beta_pdf = np.kron(y_beta_indv, np.repeat(1,deg))

denominator = np.sum(w_hat * y_beta_pdf)
denominator = 0.5025214

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
    
