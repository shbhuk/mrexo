import numpy as np
import os
from scipy.stats.mstats import mquantiles
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .mle_utils import cond_density_quantile
from .utils import load_lookup_table
from .plot import plot_r_given_m_relation, plot_m_given_r_relation

pwd = os.path.dirname(__file__)
np.warnings.filterwarnings('ignore')

def predict_from_measurement(measurement, measurement_sigma=None,
            predict = 'Mass', result_dir=None, dataset='mdwarf',
            is_posterior=False, qtl=[0.16,0.84], show_plot=False,
            use_lookup=False):
    '''
    Predict mass from given radius, or radius from mass for a single object.
    Function can be used to predict from a single measurement (w/ or w/o error), or from a posterior distribution.
    INPUT:
        measurement: Numpy array of measurement/s. Always in linear scale.
        measurement_sigma: Numpy array of radius uncertainties. Assumes symmetrical uncertainty.
                           Default : None. Always in linear scale.
        predict: The quantity that is being predicted. If = 'Mass', will give mass given input radius.
                Else, can predict radius from mass, if = 'Radius'.
        result_dir: The directory where the results of the fit are stored. Default is None.
                If None, then will either use M-dwarf or Kepler fits (supplied with package).
        dataset: If result_dir == None, then will use included fits for M-dwarfs or Kepler dataset.
                To run the M-dwarf or Kepler fit, define result_dir as None,
                and then dataset='mdwarf', or dataset='kepler'
                The Kepler dataset has been explained in Ning et al. 2018.
                The M-dwarf dataset has been explained in Kanodia et al. 2019.
        is_posterior: If the input radii is a posterior sample, is_posterior=True, else False.
                Default=False
        qtl: 2 element array or list with the quantile values that will be returned.
                Default is 0.16 and 0.84. qtl=[0.16,0.84]. If is_posterior=True, qtl will not be considered.
                Assumes that the entered qtls are symmetric about 0.5.
        show_plot: Boolean. Default=False. If True, will plot the conditional Mass - Radius relationship, and show the predicted point.
        use_lookup: If True, will try to use lookup table. If lookup table does not exist, will give warning and calculate the prediction
                using analytic method.
    OUTPUT:
        outputs: Tuple with the predicted mass (or distribution of masses if input is a posterior),
                and the quantile distribution according to the 'qtl' input parameter
    EXAMPLE:
        from mrexo import predict_from_measurement
        import os
        import numpy as np
        pwd = '~/mrexo_working/'

        #Below example predicts the mass for a radius of log10(1) Earth radii exoplanet, with no measurement
        #uncertainty from the fit results in 'M_dwarfs_deg_cv'
        result_dir = os.path.join(pwd,'M_dwarfs_deg_cv')
        predicted_mass, qtls = predict_from_measurement(measurement=1, measurement_sigma=None,
                               result_dir=result_dir, is_posterior=False, is_log=True)

        #Below example predicts the mass for a radius of log10(1) Earth radii exoplanet with uncertainty of 0.1 Earth Radii on the included Mdwarf fit.
        #Similary for Kepler dataset.
        predicted_mass, qtls = predict_from_measurement(measurement=1, measurement_sigma=0.1, result_dir=None, dataset='mdwarf', is_posterior=False,
                               is_log=True)

        #Below example predicts the radius for a mass of log10(1) Earth mass exoplanet with uncertainty of 0.1 Earth Mass on the included Mdwarf fit.
        #Similary for Kepler dataset.
        predicted_mass, qtls = predict_from_measurement(measurement=1, measurement_sigma=0.1, predict = 'radius', result_dir=None,
                               dataset='mdwarf', is_posterior=False, is_log=True)
    '''

    dataset = dataset.replace(' ', '').replace('-', '').lower()
    predict = predict.replace(' ', '').replace('-', '').lower()

    # Define the result directory.
    mdwarf_resultdir = os.path.join(pwd, 'datasets', 'M_dwarfs_20181214')
    kepler_resultdir = os.path.join(pwd, 'datasets', 'Kepler_Ning_etal_20170605')

    if result_dir == None:
        if dataset == 'mdwarf':
            result_dir = mdwarf_resultdir
        elif dataset == 'kepler':
            result_dir = kepler_resultdir

    if measurement_sigma == 0:
        measurement_sigma = None

    input_location = os.path.join(result_dir, 'input')
    output_location = os.path.join(result_dir, 'output')

    # Load the results from the directory
    Mass_min, Mass_max = np.loadtxt(os.path.join(input_location, 'Mass_bounds.txt'))
    Radius_min, Radius_max = np.loadtxt(os.path.join(input_location, 'Radius_bounds.txt'))
    weights_mle = np.loadtxt(os.path.join(output_location,'weights.txt'))
    R_points = np.loadtxt(os.path.join(output_location, 'R_points.txt'))

    degree = int(np.sqrt(len(weights_mle)))
    deg_vec = np.arange(1,degree+1)

    # Convert linear to log10.
    log_measurement = np.log10(measurement)
    if measurement_sigma:
        measurement_sigma = 0.434 * measurement_sigma / measurement


    if predict == 'mass':
        predict_min, predict_max = Mass_min, Mass_max
        measurement_min, measurement_max = Radius_min, Radius_max
        w_hat = weights_mle
        lookup_fname = 'lookup_m_given_r_interp2d.npy'

        if np.min(log_measurement) < np.log10(1.3):
            #This is from 100% iron curve of Fortney 2007; solving for
            # logM (base 10) via quadratic formula.
            Mass_iron = mass_100_percent_iron_planet(np.min(log_measurement))
            print('Mass of 100% Iron planet of {} Earth Radii = {} Earth Mass'.format(10**np.min(log_measurement), 10**Mass_iron))
    else:
        predict_min, predict_max = Radius_min, Radius_max
        measurement_min, measurement_max = Mass_min, Mass_max
        w_hat = np.reshape(weights_mle,(degree,degree)).T.flatten()
        lookup_fname = 'lookup_r_given_m_interp2d.npy'

    ########################################################

    # Check if single measurement or posterior distribution.
    if is_posterior == False:

        # Use 0.5 to find the median
        lookup_flag = None
        if use_lookup == True:
            try:
                lookup = load_lookup_table(os.path.join(output_location,lookup_fname))
                if not measurement_sigma:
                    predicted_median = lookup(0.5, log_measurement)
                    predicted_qtl = lookup(qtl, log_measurement)
                else:
                    n_art = 10000
                    artificial_posterior = np.random.normal(loc = log_measurement, scale = measurement_sigma, size = n_art)
                    qtls = np.random.uniform(size = n_art)
                    predicted_posteriors = [lookup(qtls[i], artificial_posterior[i]) for i in range(n_art)]

                    prob = [0.5]
                    prob.extend(qtl)
                    result = mquantiles(predicted_posteriors, prob=prob, axis=0, alphap=1, betap=1).data
                    predicted_median = result[0]
                    predicted_qtl = result[1:]

                lookup_flag = 1
            except FileNotFoundError:
                print('Error: Trying to use lookup table when it does not exist. Run script to generate lookup table or set use_lookup = False.')
                lookup_flag = None

        if not lookup_flag:
            predicted_value = cond_density_quantile(y=log_measurement, y_std=measurement_sigma, y_max=measurement_max,
                                                        y_min=measurement_min, x_max=predict_max, x_min=predict_min,
                                                        deg=degree, deg_vec = deg_vec,
                                                        w_hat=w_hat, qtl=np.insert(np.array(qtl),0,0.5))
            predicted_median = predicted_value[2][0]
            predicted_qtl = predicted_value[2][1:]

        outputs = [predicted_median, np.array(predicted_qtl)]

        if show_plot == True:

            if np.size(qtl)==2:
                predicted_lower_quantile, predicted_upper_quantile = predicted_qtl
            else:
                # If finding multiple quantiles, do not plot errorbar on predicted value in plot
                predicted_lower_quantile, predicted_upper_quantile = predicted_median, predicted_median

            if predict == 'mass':
                fig, ax, handles = plot_m_given_r_relation(result_dir=result_dir)
                ax.plot(R_points, mass_100_percent_iron_planet(R_points), 'k')
                handles.append(Line2D([0], [0], color='k',  label=r'100$\%$ Iron planet'))
            else:
                fig, ax, handles = plot_r_given_m_relation(result_dir=result_dir)

            yerr = np.array([[predicted_median - predicted_lower_quantile, predicted_upper_quantile - predicted_median]]).T

            plt.hlines(predicted_median, measurement_min, measurement_max, linestyle = 'dashed', colors = 'darkgrey')
            plt.vlines(log_measurement, predict_min, predict_max,linestyle = 'dashed', colors = 'darkgrey')

            ax.errorbar(x=log_measurement, y=predicted_median, xerr=measurement_sigma,
                        yerr=yerr,fmt='o', color = 'green')
            handles.append(Line2D([0], [0], color='green', marker='o',  label='Predicted value'))
            plt.legend(handles=handles)
            plt.show()

    ###########################################################

    elif is_posterior == True:

        n = np.size(measurement)
        random_quantile = np.zeros(n)

        lookup_flag = None
        random_quantile = np.zeros((n))

        if use_lookup == True:
            try:
                lookup = load_lookup_table(os.path.join(output_location,lookup_fname))
                lookup_flag = 1
                for i in range(0,n):
                    qtl_check = np.random.random()
                    random_quantile[i] = lookup(qtl_check, log_measurement[i])
            except FileNotFoundError:
                print('Error: Trying to use lookup table when it does not exist. Run script to generate lookup table or set use_lookup = False.')

        if not lookup_flag:
            for i in range(0,n):
                qtl_check = np.random.random()
                results = cond_density_quantile(y=log_measurement[i], y_std=None, y_max=measurement_max, y_min=measurement_min,
                                                        x_max=predict_max, x_min=predict_min, deg=degree, deg_vec = deg_vec,
                                                        w_hat=w_hat, qtl=[qtl_check])

                random_quantile[i] = results[2][0]

        outputs = random_quantile

        if show_plot == True:

            if predict == 'mass':
                fig, ax, handles = plot_m_given_r_relation(result_dir=result_dir)
                ax.plot(R_points, mass_100_percent_iron_planet(R_points), 'k')
                handles.append(Line2D([0], [0], color='k',  label='100% Iron planet'))

            else:
                fig, ax, handles = plot_r_given_m_relation(result_dir=result_dir)

            if np.size(qtl)==2:
                predicted_lower_quantile, predicted_upper_quantile = predicted_qtl
                output_qtl =  mquantiles(outputs, prob=[0.5,qtl],axis=0,alphap=1,betap=1).data
                measurement_qtl = mquantiles(log_measurement ,prob=[0.5,qtl],axis=0,alphap=1,betap=1).data
            else:
                # If finding multiple quantiles, do not plot errorbar on predicted value in plot
                output_qtl =  mquantiles(outputs, prob=[0.5,0.5],axis=0,alphap=1,betap=1).data
                measurement_qtl = mquantiles(log_measurement ,prob=[0.5,0.5],axis=0,alphap=1,betap=1).data


            plt.hlines(output_qtl[0], measurement_min, measurement_max, linestyle = 'dashed', colors = 'darkgrey')
            plt.vlines(measurement_qtl[0], predict_min, predict_max,linestyle = 'dashed', colors = 'darkgrey')
            plt.plot(log_measurement,outputs,'g.',markersize = 9)

            ax.errorbar(x=measurement_qtl[0], y=output_qtl[0], xerr=np.abs(measurement_qtl[0] - measurement_qtl[1]),
                        yerr=np.abs(output_qtl[0] - output_qtl[1]),fmt='o', color = 'green')
            handles.append(Line2D([0], [0], color='green', marker='o',  label='Predicted value'))
            plt.legend(handles=handles)
            plt.show()


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



def generate_lookup_table(predict_quantity = 'Mass', result_dir = None):
    '''
    Generate lookup table size 1000x1000 to make the prediction function faster.
    Then in predict_from_measurement() set use_lookup = True.
    INPUTS:
        predict_quantity: To predict mass from radius, set to 'mass'. To go the other way,
                          set to 'radius'. Default = 'Mass'
        result_dir: Directory generated by the fitting procedure.

    OUTPUTS: The generated lookup table is saved in /result_dir/output/ in the form
             of a .txt file as well as a .npy file which has the 2D interpolated version
             of the lookup table.

    EXAMPLE:
        ## To generate lookup table to get mass from radius
        from mrexo.predict import generate_lookup_table
        kepler_result = '/storage/home/s/szk381/work/mrexo/mrexo/datasets/Kepler_Ning_etal_20170605'

        generate_lookup_table(result_dir = kepler_result, predict_quantity = 'Mass')

    '''

    predict_quantity = predict_quantity.replace(' ', '').replace('-', '').lower()

    input_location = os.path.join(result_dir, 'input')
    output_location = os.path.join(result_dir, 'output')
    Mass_min, Mass_max = np.loadtxt(os.path.join(input_location, 'Mass_bounds.txt'))
    Radius_min, Radius_max = np.loadtxt(os.path.join(input_location, 'Radius_bounds.txt'))

    lookup_grid_size = 1000

    lookup_table = np.zeros((lookup_grid_size, lookup_grid_size))
    qtl_steps = np.linspace(0,1,lookup_grid_size)

    if predict_quantity == 'mass':
        search_steps = np.linspace(Radius_min, Radius_max, lookup_grid_size)
        fname = 'lookup_m_given_r'
        comment = 'Lookup table for predicting log(Mass) given log(Radius) and certain quantile.'
    else:
        search_steps = np.linspace(Mass_min, Mass_max, lookup_grid_size)
        fname = 'lookup_r_given_m'
        comment = 'Lookup table for predicting log(Radius) given log(Mass) and certain quantile.'


    for i in range(0,lookup_grid_size):
        lookup_table[i,:] = predict_from_measurement(measurement = search_steps[i], qtl = qtl_steps,
                            result_dir = result_dir, is_log = True, predict = predict_quantity)[1]
        if i%100==0:
            print(i)

    np.savetxt(os.path.join(output_location,fname+'.txt'), lookup_table, comments='#', header=comment)

    interp = interp2d(qtl_steps, search_steps, lookup_table)
    np.save(os.path.join(output_location,fname+'_interp2d.npy'), interp)
