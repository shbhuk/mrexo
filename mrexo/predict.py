import numpy as np
import os
from scipy.stats.mstats import mquantiles
from scipy.interpolate import interp1d

from .mle_utils import cond_density_quantile
from .plot import plot_r_given_m_relation, plot_m_given_r_relation

pwd = os.path.dirname(__file__)
np.warnings.filterwarnings('ignore')

def predict_m_given_r(Radius,  Radius_sigma=None, result_dir=None, dataset='mdwarf',
                      posterior_sample=False, qtl=[0.16,0.84], islog=False, showplot=False):
    '''
    Predict mass from given radius.
    Given radius can be a single measurement, with or without error, or can also be a posterior distribution of radii.
    INPUT:
        Radius: Numpy array of radius measurements.
        Radius_sigma: Numpy array of radius uncertainties. Assumes symmetrical uncertainty. Default : None
        result_dir: The directory where the results of the fit are stored. Default is None.
                If None, then will either use M-dwarf or Kepler fits (supplied with package).
        dataset: If result_dir == None, then will use included fits for M-dwarfs or Kepler dataset.
                To run the M-dwarf or Kepler set, define result_dir as None,
                and then dataset='mdwarf', or dataset='kepler'
                The Kepler dataset has been explained in Ning et al. 2018.
                The M-dwarf dataset has been explained in Kanodia et al. 2019.
        posterior_sample: If the input radii is a posterior sample, posterior_sample=True, else False.
                Default=False
        qtl: 2 element array or list with the quantile values that will be returned.
                Default is 0.16 and 0.84. qtl=[0.16,0.84]
        islog: Whether the radius given is in log scale or not.
                Default is False. The Radius_sigma is always in original units
        showplot: Boolean. Default=False. If True, will plot the conditional Mass - Radius relationship, and show the predicted point.
    OUTPUT:
        outputs: Tuple with the predicted mass (or distribution of masses if input is a posterior),
                and the quantile distribution according to the 'qtl' input parameter
    EXAMPLE:
        #Below example predicts the mass for a radius of log10(1) Earth radii exoplanet, with no measurement uncertainty from the fit results in 'M_dwarfs_deg_cv'
        from mrexo import predict_m_given_r
        import os
        import numpy as np
        pwd = '~/mrexo_working/'
        result_dir = os.path.join(pwd,'M_dwarfs_deg_cv')
        predicted_mass, lower_qtl_mass, upper_qtl_mass = predict_m_given_r(Radius=1, Radius_sigma=None, result_dir=result_dir, posterior_sample=False, islog=True)
        #Below example predicts the mass for a radius of log10(1) Earth radii exoplanet with uncertainty of 0.1 Earth Radii on the included Mdwarf fit. Similary for Kepler dataset.
        predicted_mass, lower_qtl_mass, upper_qtl_mass = predict_m_given_r(Radius=1, Radius_sigma=0.1, result_dir=None, dataset='mdwarf', posterior_sample=False, islog=True)
    '''

    dataset = dataset.replace(' ', '').replace('-', '').lower()

    # Define the result directory.
    mdwarf_resultdir = os.path.join(pwd, 'datasets', 'M_dwarfs_20181214')
    kepler_resultdir = os.path.join(pwd, 'datasets', 'Kepler_Ning_etal_20170605')

    if result_dir == None:
        if dataset == 'mdwarf':
            result_dir = mdwarf_resultdir
        elif dataset == 'kepler':
            result_dir = kepler_resultdir

    print(result_dir)

    input_location = os.path.join(result_dir, 'input')
    output_location = os.path.join(result_dir, 'output')

    # Load the results from the directory
    Mass_min, Mass_max = np.loadtxt(os.path.join(input_location, 'Mass_bounds.txt'))
    Radius_min, Radius_max = np.loadtxt(os.path.join(input_location, 'Radius_bounds.txt'))
    weights_mle = np.loadtxt(os.path.join(output_location,'weights.txt'))

    R_points = np.loadtxt(os.path.join(output_location, 'R_points.txt'))

    M_cond_R_boot = np.loadtxt(os.path.join(output_location, 'M_cond_R_boot.txt'))
    lower_boot, upper_boot = mquantiles(M_cond_R_boot,prob=[0.16, 0.84],axis=0,alphap=1,betap=1).data

    degree = int(np.sqrt(len(weights_mle)))
    deg_vec = np.arange(1,degree+1)

    # Convert the radius measurement to log scale.
    if islog == False:
        logRadius = np.log10(Radius)
        if Radius_sigma:
            Radius_sigma = 0.434 * Radius_sigma / Radius
    else:
        logRadius = Radius
    

    # Check if single measurement or posterior distribution.
    if posterior_sample == False:
        if logRadius < np.log10(1.3):
            #This is from 100% iron curve of Fortney 2007; solving for logM (base 10) via quadratic formula.
            Mass_iron = mass_100_percent_iron_planet(logRadius)
            print('Mass of 100% Iron planet of {} Earth Radii = {} Earth Mass'.format(10**logRadius, 10**Mass_iron))
            
        predicted_value = cond_density_quantile(y=logRadius, y_std=Radius_sigma, y_max=Radius_max, y_min=Radius_min,
                                                      x_max=Mass_max, x_min=Mass_min, deg=degree, deg_vec = deg_vec,
                                                      w_hat=weights_mle, qtl=qtl)
        predicted_mean = predicted_value[0]
        predicted_lower_quantile, predicted_upper_quantile = predicted_value[2]

        outputs = [predicted_mean,predicted_lower_quantile,predicted_upper_quantile]

        if showplot == True:
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D

            fig, ax, handles = plot_m_given_r_relation(result_dir=result_dir)
            plt.hlines(predicted_mean, Radius_min, Radius_max, linestyle = 'dashed', colors = 'darkgrey')
            plt.vlines(logRadius, Mass_min, Mass_max,linestyle = 'dashed', colors = 'darkgrey')
            ax.errorbar(x=logRadius, y=predicted_mean, xerr=Radius_sigma,
                        yerr=[[predicted_mean - predicted_lower_quantile, predicted_upper_quantile - predicted_mean]],
                        fmt='o', color = 'green')
            ax.plot(R_points, mass_100_percent_iron_planet(R_points), 'k')
            handles.append(Line2D([0], [0], color='green', marker='o',  label='Predicted value'))
            handles.append(Line2D([0], [0], color='k',  label='100% Iron planet'))
            plt.legend(handles=handles)


    elif posterior_sample == True:

        if np.min(logRadius) < np.log10(1.3):
            #This is from 100% iron curve of Fortney 2007; solving for logM (base 10) via quadratic formula.
            Mass_iron = mass_100_percent_iron_planet(np.min(logRadius))
            print('Mass of 100% Iron planet of {} Earth Radii = {} Earth Mass'.format(10**np.min(logRadius), 10**Mass_iron))

            
        n = np.size(Radius)
        mean_sample = np.zeros(n)
        random_quantile = np.zeros(n)

        if n != np.size(Radius_sigma):
            Radius_sigma = np.repeat(None,n)

        for i in range(0,n):
            qtl_check = np.random.random()
            results = cond_density_quantile(y=logRadius[i], y_std=Radius_sigma[i], y_max=Radius_max, y_min=Radius_min,
                                                      x_max=Mass_max, x_min=Mass_min, deg=degree, deg_vec = deg_vec,
                                                      w_hat=weights_mle, qtl=[qtl_check,0.5])

            mean_sample[i] = results[0]
            random_quantile[i] = results[2][0]

        outputs = random_quantile


        if showplot == True:
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D

            r_q =  mquantiles(logRadius, prob=[0.16, 0.5, 0.84],axis=0,alphap=1,betap=1).data
            m_q = mquantiles(outputs ,prob=[0.16, 0.5, 0.84],axis=0,alphap=1,betap=1).data

            fig, ax, handles = plot_m_given_r_relation(result_dir=result_dir)
            plt.hlines(m_q[1], Radius_min, Radius_max, linestyle = 'dashed', colors = 'darkgrey')
            plt.vlines(r_q[1], Mass_min, Mass_max,linestyle = 'dashed', colors = 'darkgrey')
            plt.plot(logRadius,outputs,'g.',markersize = 9)
            ax.errorbar(x=r_q[1], y=m_q[1], xerr=r_q[1] - r_q[0],  yerr=m_q[1] - m_q[0], fmt='o', color = 'green')
            ax.plot(R_points, mass_100_percent_iron_planet(R_points), 'k')
            handles.append(Line2D([0], [0], color='green', marker='o',  label='Predicted value'))
            handles.append(Line2D([0], [0], color='k',  label='100% Iron planet'))
            plt.legend(handles=handles)


    if islog:
        return outputs
    else:
        return [10**x for x in outputs]



def predict_r_given_m(Mass,  Mass_sigma=None, result_dir=None, dataset='mdwarf',
                      posterior_sample=False, qtl=[0.16,0.84], islog=False, showplot=False):
    '''
    Predict radius from given mass.
    Given mass can be a single measurement, with or without error, or can also be a posterior distribution of mass.
    INPUT:
        Mass: Numpy array of mass measurements.
        Mass_sigma: Numpy array of mass uncertainties. Assumes symmetrical uncertainty. Default : None
        result_dir: The directory from where the results of the fit are read in. Default is None.
                If None, then will either use M-dwarf or Kepler fits (supplied with package).
        dataset: If result_dir == None, then will use included fits for M-dwarfs or Kepler dataset.
                To run the M-dwarf or Kepler set, define result_dir as None,
                and then dataset='mdwarf', or dataset='kepler'
                The Kepler dataset has been explained in Ning et al. 2018.
                The M-dwarf dataset has been explained in Kanodia et al. 2019.
        posterior_sample: If the input mass is a posterior sample, posterior_sample=True, else False.
                Default=False
        qtl: 2 element array or list with the quantile values that will be returned.
                Default is 0.16 and 0.84. qtl=[0.16,0.84]
        islog: Whether the radius given is in log scale or not.
                Default is False. The Radius_sigma is always in original units
        showplot: Boolean. Default=False. If True, will plot the conditional Mass - Radius relationship, and show the predicted point.
    OUTPUT:
        outputs: Tuple with the predicted radius (or distribution of radii if input is a posterior),
                and the quantile distribution according to the 'qtl' input parameter
    EXAMPLE:
        #Below example predicts the radius for a mass of log10(1) Earth radii exoplanet, with no measurement uncertainty from the fit results in 'M_dwarfs_deg_cv'
        from mrexo import predict_r_given_m
        import os
        import numpy as np
        pwd = '~/mrexo_working/'
        result_dir = os.path.join(pwd,'M_dwarfs_deg_cv')
    '''

    dataset = dataset.replace(' ', '').replace('-', '').lower()

    # Define the result directory.
    mdwarf_resultdir = os.path.join(pwd, 'datasets', 'M_dwarfs_20181214')
    kepler_resultdir = os.path.join(pwd, 'datasets', 'Kepler_Ning_etal_20170605')

    if result_dir == None:
        if dataset == 'mdwarf':
            result_dir = mdwarf_resultdir
        elif dataset == 'kepler':
            result_dir = kepler_resultdir

    print(result_dir)

    input_location = os.path.join(result_dir, 'input')
    output_location = os.path.join(result_dir, 'output')

    # Load the results from the directory
    Mass_min, Mass_max = np.loadtxt(os.path.join(input_location, 'Mass_bounds.txt'))
    Radius_min, Radius_max = np.loadtxt(os.path.join(input_location, 'Radius_bounds.txt'))
    weights_mle = np.loadtxt(os.path.join(output_location,'weights.txt'))

    R_cond_M_boot = np.loadtxt(os.path.join(output_location, 'R_cond_M_boot.txt'))
    lower_boot, upper_boot = mquantiles(R_cond_M_boot,prob=[0.16, 0.84],axis=0,alphap=1,betap=1).data

    degree = int(np.sqrt(len(weights_mle)))
    deg_vec = np.arange(1,degree+1)

    # Convert the mass measurement to log scale.
    if islog == False:
        logMass = np.log10(Mass)
        if Mass_sigma:
            Mass_sigma = 0.434 * Mass_sigma / Mass
    else:
        logMass = Mass

    # Check if single measurement or posterior distribution.
    if posterior_sample == False:
        predicted_value = cond_density_quantile(y=logMass, y_std=Mass_sigma, y_max=Mass_max, y_min=Mass_min,
                                                      x_max=Radius_max, x_min=Radius_min, deg=degree, deg_vec = deg_vec,
                                                      w_hat=np.reshape(weights_mle,(degree,degree)).T.flatten(), qtl=qtl)
        predicted_mean = predicted_value[0]
        predicted_lower_quantile, predicted_upper_quantile = predicted_value[2]

        outputs = [predicted_mean,predicted_lower_quantile,predicted_upper_quantile]


        if showplot == True:
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D

            fig, ax, handles = plot_r_given_m_relation(result_dir=result_dir)
            plt.hlines(predicted_mean, Mass_min, Mass_max, linestyle = 'dashed', colors = 'darkgrey')
            plt.vlines(logMass, Radius_min, Radius_max, linestyle = 'dashed', colors = 'darkgrey')
            ax.errorbar(x=logMass, y=predicted_mean, xerr=Mass_sigma,
                        yerr=[[predicted_mean - predicted_lower_quantile, predicted_upper_quantile - predicted_mean]],
                        fmt='o', color = 'green')
            handles.append(Line2D([0], [0], color='green', marker='o',  label='Predicted value'))
            plt.legend(handles=handles)



    elif posterior_sample == True:

        n = np.size(Mass)
        mean_sample = np.zeros(n)
        random_quantile = np.zeros(n)
        
        if n != np.size(Mass_sigma):
            Mass_sigma = np.repeat(None,n)

        for i in range(0,n):
            qtl_check = np.random.random()
            results = cond_density_quantile(y=logMass[i], y_std=Mass_sigma[i], y_max=Mass_max, y_min=Mass_min,
                                                      x_max=Radius_max, x_min=Radius_min, deg=degree, deg_vec = deg_vec,
                                                      w_hat=np.reshape(weights_mle,(degree,degree)).T.flatten(), qtl=[qtl_check,0.5])

            mean_sample[i] = results[0]
            random_quantile[i] = results[2][0]

        outputs = random_quantile

        if showplot == True:
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D

            m_q =  mquantiles(logMass, prob=[0.16, 0.5, 0.84],axis=0,alphap=1,betap=1).data
            r_q = mquantiles(outputs ,prob=[0.16, 0.5, 0.84],axis=0,alphap=1,betap=1).data

            fig, ax, handles = plot_r_given_m_relation(result_dir=result_dir)
            plt.plot(logMass,outputs,'g.',markersize = 9)
            plt.hlines(r_q[1], Mass_min, Mass_max, linestyle = 'dashed', colors = 'darkgrey')
            plt.vlines(m_q[1], Radius_min, Radius_max, linestyle = 'dashed', colors = 'darkgrey')
            ax.errorbar(y=r_q[1], x=m_q[1], yerr=r_q[1] - r_q[0],  xerr=m_q[1] - m_q[0],
                        fmt='o', color = 'green')
            handles.append(Line2D([0], [0], color='green', marker='o',  label='Predicted value'))
            plt.legend(handles=handles)



    if islog:
        return outputs
    else:
        return [10**x for x in outputs]


def mass_100_percent_iron_planet(logRadius):
    '''
    This is from 100% iron curve of Fortney 2007; solving for logM (base 10) via quadratic formula.
    INPUT:
        logRadius : Radius of the planet in log10 units
    OUTPUT:
        logMass: Mass in log10 units for a 100% iron planet of given radius
    '''

    Mass_iron = (-0.4938 + np.sqrt(0.4938**2-4*0.0975*(0.7932-10**(logRadius))))/(2*0.0975)
    return Mass_iron

def find_mass_probability_distribution_function(R_check, Radius_min, Radius_max, Mass_max, Mass_min, weights_mle, weights_boot, degree, deg_vec, M_points, islog = True):
    '''
    
    '''  

    if islog == False:
        R_check = np.log10(R_check)
        
    n_quantiles = 200
    qtl = np.linspace(0,1.0, n_quantiles)

    results = cond_density_quantile(y=R_check, y_std=None, y_max=Radius_max, y_min=Radius_min,
                                                      x_max=Mass_max, x_min=Mass_min, deg=degree, deg_vec=deg_vec,
                                                      w_hat=weights_mle, qtl=qtl)                                                      

    interpolated_qtls = interp1d(results[2], qtl)(M_points)

    # Conditional_plot. PDF is derivative of CDF
    pdf_interp = np.diff(interpolated_qtls) / np.diff(M_points)
   
    n_boot = np.shape(weights_boot)[0]
    n_boot = 50
    pdf_boots = np.zeros((n_boot, len(M_points) - 1))
    
    for i in range(0, n_boot):
        weight = weights_boot[i,:]
        results_boot = cond_density_quantile(y=R_check, y_std=None, y_max=Radius_max, y_min=Radius_min,
                                                        x_max=Mass_max, x_min=Mass_min, deg=degree, deg_vec=deg_vec,
                                                        w_hat=weight, qtl=qtl)
        interpolated_qtls = interp1d(results_boot[2], qtl)(M_points)
        pdf_boots[i] = np.diff(interpolated_qtls) / np.diff(M_points)
        print(i)
        
    lower_boot, upper_boot = mquantiles(pdf_boots ,prob=[0.16, 0.84],axis=0,alphap=1,betap=1).data
    
    return pdf_interp, lower_boot, upper_boot


def find_radius_probability_distribution_function(M_check, Mass_max, Mass_min, Radius_min, Radius_max, weights_mle, weights_boot, degree, deg_vec, R_points, islog = True):
    '''
    
    '''  
    
    if islog == False:
        M_check = np.log10(M_check)
        
    n_quantiles = 200
    qtl = np.linspace(0,1.0, n_quantiles)

    results = cond_density_quantile(y=M_check, y_std=None, y_max=Mass_max, y_min=Mass_min,
                                                      x_max=Radius_max, x_min=Radius_min, deg=degree, deg_vec=deg_vec,
                                                      w_hat=weights_mle, qtl=qtl)                                                      

    interpolated_qtls = interp1d(results[2], qtl)(R_points)

    # Conditional_plot. PDF is derivative of CDF
    pdf_interp = np.diff(interpolated_qtls) / np.diff(R_points)
   
    n_boot = np.shape(weights_boot)[0]
    n_boot = 50
    pdf_boots = np.zeros((n_boot, len(R_points) - 1))
    
    for i in range(0, n_boot):
        weight = weights_boot[i,:]
        results_boot = cond_density_quantile(y=M_check, y_std=None, y_max=Mass_max, y_min=Mass_min,
                                                        x_max=Radius_max, x_min=Radius_min, deg=degree, deg_vec=deg_vec,
                                                        w_hat=weight, qtl=qtl)
        interpolated_qtls = interp1d(results_boot[2], qtl)(R_points)
        pdf_boots[i] = np.diff(interpolated_qtls) / np.diff(R_points)
        print(i)
        
    lower_boot, upper_boot = mquantiles(pdf_boots ,prob=[0.16, 0.84],axis=0,alphap=1,betap=1).data
    
    return pdf_interp, lower_boot, upper_boot
