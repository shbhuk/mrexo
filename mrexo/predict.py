import numpy as np
import os
from scipy.stats.mstats import mquantiles
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from multiprocessing import Pool,cpu_count

from .mle_utils import cond_density_quantile
from .utils import _load_lookup_table
from .plot import plot_x_given_y_relation, plot_y_given_x_relation

pwd = os.path.dirname(__file__)
np.warnings.filterwarnings('ignore')

def predict_from_measurement(measurement, measurement_sigma=np.nan,
            predict = 'mass', result_dir=None, dataset='mdwarf',
            is_posterior=False, qtl=[0.16,0.84], show_plot=False,
            use_lookup=False):
    """
    Predict Qty Y from Qty X or X from Y for a single object on the basis of the XY nonparametric fit.
    Function can be used to predict from a single measurement (w/ or w/o error), or from a posterior distribution.
    \nINPUTS:
        measurement: Numpy array of measurement/s. Always in linear scale.
        measurement_sigma: Numpy array of radius uncertainties. Assumes
            symmetrical uncertainty. Default : None. Always in linear scale.
        predict: The quantity that is being predicted.
                Specify based on Xlabel and Ylabel used for fitting.
                If = 'Mass', will give mass given input radius.
                Else, can predict radius from mass, if = 'Radius'.
        result_dir: The directory where the results of the fit are stored.
                Default is None. If None, then will either use M-dwarf or
                Kepler fits (supplied with package).
        dataset: If result_dir == None, then will use included fits for
                M-dwarfs or Kepler dataset.
                To run the M-dwarf or Kepler fit, define result_dir as None,
                and then dataset='mdwarf', or dataset='kepler'
                The Kepler dataset has been explained in Ning et al. 2018.
                The M-dwarf dataset has been explained in Kanodia et al. 2019.
        is_posterior: If the input radii is a posterior sample,
                is_posterior=True, else False.
                Default=False
        qtl: 2 element array or list with the quantiles that will be returned.
                Default is 0.16 and 0.84. qtl=[0.16,0.84].
                If is_posterior=True, qtl will not be considered.
        show_plot: Boolean. Default=False.
                If True, will plot the conditional Mass - Radius relationship,
                and show the predicted point.
        use_lookup: If True, will try to use lookup table.
                If lookup table does not exist, will give warning and
                calculate the prediction using analytic method.
                Can only be used for posterior prediction.
    OUTPUTS:

        outputs: Tuple with the predicted mass
                (or distribution of masses if input is a posterior),
                and the quantile distribution according to
                the 'qtl' input parameter.
        iron_planet: Corresponding predicted quantity for a 100% Iron planet.
                Example: If predicting mass from radius, iron_planet will be
                the mass of a 100% Iron planet for the radius. Similarly for
                radius predictions from mass. This should be used as a reality
                check, in the small planet mass regime,where the uncertainties
                dominate, and can give unphysical results.
    EXAMPLE:

        from mrexo import predict_from_measurement
        import os
        import numpy as np
        pwd = '~/mrexo_working/'

        #Below example predicts the mass for a radius of log10(1) Earth radii
        #exoplanet, with no measurement uncertainty from
        #the fit results in 'M_dwarfs_deg_cv'

        result_dir = os.path.join(pwd,'M_dwarfs_deg_cv')
        predicted_mass, qtls = predict_from_measurement(measurement=1,
                               measurement_sigma=None,
                               result_dir=result_dir,
                            is_posterior=False, is_log=True)

        #Below example predicts the mass for a radius of log10(1) Earth radii
        #exoplanet with uncertainty of 0.1 Earth Radii on the included Mdwarf fit.
        predicted_mass, qtls = predict_from_measurement(
                                measurement=1, measurement_sigma=0.1,
                                result_dir=None, dataset='mdwarf',
                                is_posterior=False,
                                is_log=True)

        #Below example predicts the radius for a mass of log10(1) Earth mass
        #exoplanet with uncertainty of 0.1 Earth Mass on the included Mdwarf fit.
        #Similary for Kepler dataset.
        predicted_mass, qtls = predict_from_measurement(
                                measurement=1, measurement_sigma=0.1,
                                predict = 'radius',
                                result_dir=None, dataset='mdwarf',
                                is_posterior=False, is_log=True)
    """

    dataset = dataset.replace(' ', '').replace('-', '').lower()
    predict = predict.replace(' ', '').replace('-', '').lower()

    # Define the result directory.
    mdwarf_resultdir = os.path.join(pwd, 'datasets', 'M_dwarfs_20200520')
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
    aux_output_location = os.path.join(output_location, 'other_data_products')

    with open(os.path.join(aux_output_location, 'AxesLabels.txt'), 'r') as f:
        LabelDictionary = eval(f.read())

    # Load the results from the directory
    Y_min, Y_max = np.loadtxt(os.path.join(input_location, 'Y_bounds.txt'))
    X_min, X_max = np.loadtxt(os.path.join(input_location, 'X_bounds.txt'))
    weights_mle = np.loadtxt(os.path.join(output_location,'weights.txt'))
    X_points = np.loadtxt(os.path.join(output_location, 'X_points.txt'))
    Y_points = np.loadtxt(os.path.join(output_location, 'Y_points.txt'))
    Y_label = LabelDictionary['Y_label'].replace(' ', '').lower()
    X_label = LabelDictionary['X_label'].replace(' ', '').lower()

    degree = int(np.sqrt(len(weights_mle)))
    deg_vec = np.arange(1,degree+1)

    # Convert linear to log10.
    log_measurement = np.log10(measurement)
    if measurement_sigma:
        log_measurement_sigma = 0.434 * measurement_sigma / measurement
    else:
        log_measurement_sigma = None


    if predict==Y_label:
        predict_min, predict_max = Y_min, Y_max
        measurement_min, measurement_max = X_min, X_max
        w_hat = weights_mle
        lookup_fname = 'lookup_y_given_x_interp2d.npy'

    elif predict==X_label:
        predict_min, predict_max = X_min, X_max
        measurement_min, measurement_max = Y_min, Y_max
        w_hat = np.reshape(weights_mle,(degree,degree)).T.flatten()
        lookup_fname = 'lookup_x_given_y_interp2d.npy'
    else:
        print("predict keyword does not match X or Y label")
        raise ValueError


    if predict == 'mass':
        Mass_iron = mass_100_percent_iron_planet(np.min(log_measurement))
        iron_planet = Mass_iron

        if np.min(log_measurement) < np.log10(1.3):
            print('Mass of 100% Iron planet of {} Earth Radii = {} Earth Mass (Fortney, Marley and Barnes 2007)'.format(10**np.min(log_measurement), 10**Mass_iron))

    elif predict == 'radius':
        Radius_iron = radius_100_percent_iron_planet(np.min(log_measurement))
        print('Radius of 100% Iron planet of {} Earth Mass = {} Earth Radii (Fortney, Marley and Barnes 2007)'.format(10**np.min(log_measurement), 10**Radius_iron))
        iron_planet = Radius_iron

    else:
        iron_planet = np.nan


    ########################################################

    # Check if single measurement or posterior distribution.
    if is_posterior == False:

            predicted_value = cond_density_quantile(a=log_measurement, a_std=log_measurement_sigma, a_max=measurement_max,
                                                        a_min=measurement_min, b_max=predict_max, b_min=predict_min,
                                                        deg=degree, deg_vec = deg_vec,
                                                        w_hat=w_hat, qtl=np.insert(np.array(qtl),0,0.5))

            predicted_median = predicted_value[2][0]
            predicted_qtl = predicted_value[2][1:]

            outputs = [predicted_median, np.array(predicted_qtl), iron_planet]

            if show_plot == True:

                if np.size(qtl)==2:
                    predicted_lower_quantile, predicted_upper_quantile = predicted_qtl
                else:
                    # If finding multiple quantiles, do not plot errorbar on predicted value in plot
                    predicted_lower_quantile, predicted_upper_quantile = predicted_median, predicted_median

                if predict == Y_label:
                    fig, ax, handles = plot_y_given_x_relation(result_dir=result_dir)
                else:
                    fig, ax, handles = plot_x_given_y_relation(result_dir=result_dir)

                if predict == 'mass':
                    ax.plot(10**X_points, 10**mass_100_percent_iron_planet(X_points), 'k')
                    handles.append(Line2D([0], [0], color='k',  label=r'100$\%$ Iron planet'))

                yerr = np.array([[10**predicted_median - 10**predicted_lower_quantile, 10**predicted_upper_quantile - 10**predicted_median]]).T

                plt.hlines(10**predicted_median, 10**measurement_min, 10**measurement_max, linestyle = 'dashed', colors = 'darkgrey')
                plt.vlines(10**log_measurement, 10**predict_min, 10**predict_max,linestyle = 'dashed', colors = 'darkgrey')
                ax.errorbar(x=measurement, y=10**predicted_median, xerr=measurement_sigma,
                            yerr=yerr,fmt='o', color = 'green')
                handles.append(Line2D([0], [0], color='green', marker='o',  label='Predicted value'))
                plt.legend(handles=handles)
                plt.show(block=False)

    ###########################################################

    elif is_posterior==True:

        n = np.size(measurement)
        random_quantile = np.zeros(n)
        lookup_flag = None

        if use_lookup==True:
            try:
                lookup = _load_lookup_table(os.path.join(output_location,lookup_fname))
                lookup_flag = 1
                for i in range(0,n):
                    qtl_check = np.random.random()
                    random_quantile[i] = lookup(qtl_check, log_measurement[i])
            except FileNotFoundError:
                print('Error: Trying to use lookup table when it does not exist. Run script to generate lookup table or set use_lookup = False.')

        if not lookup_flag:
            for i in range(0,n):
                qtl_check = np.random.random()

                results = cond_density_quantile(a=log_measurement[i], a_std=None, a_max=measurement_max, a_min=measurement_min,
                                                        b_max=predict_max, b_min=predict_min, deg=degree, deg_vec = deg_vec,
                                                        w_hat=w_hat, qtl=[qtl_check])

                random_quantile[i] = results[2][0]

        outputs = [random_quantile]

        if show_plot == True:

            if predict==Y_label:
                fig, ax, handles = plot_y_given_x_relation(result_dir=result_dir)
            else:
                fig, ax, handles = plot_x_given_y_relation(result_dir=result_dir)

            if predict == 'mass':
                ax.plot(10**X_points, 10**mass_100_percent_iron_planet(X_points), 'k')
                handles.append(Line2D([0], [0], color='k',  label=r'100$\%$ Iron planet'))

            # Check if need this if-else block
            if np.size(qtl)==2:
                # predicted_lower_quantile, predicted_upper_quantile = predicted_qtl
                output_qtl =  mquantiles(outputs[:-1], prob=[0.5, qtl[0], qtl[1]],axis=0,alphap=1,betap=1).data
                measurement_qtl = mquantiles(log_measurement, prob=[0.5, qtl[0], qtl[1]],axis=0,alphap=1,betap=1).data
            else:
                # If finding multiple quantiles, do not plot errorbar on predicted value in plot
                output_qtl =  mquantiles(outputs[:-1], prob=[0.5,0.5],axis=0,alphap=1,betap=1).data
                measurement_qtl = mquantiles(log_measurement, prob=[0.5, 0.5],axis=0,alphap=1,betap=1).data

            # plt.hlines(10**output_qtl[0], 10**measurement_min, 10**measurement_max, linestyle = 'dashed', colors = 'darkgrey')
            # plt.vlines(10**measurement_qtl[0], 10**predict_min, 10**predict_max,linestyle = 'dashed', colors = 'darkgrey')

            plt.plot(measurement,10**random_quantile,'g.',markersize = 8, alpha = 0.3)

            # ax.errorbar(x=10**measurement_qtl[0], y=10**output_qtl[0], xerr=np.abs(10**measurement_qtl[0] - 10**measurement_qtl[1]),
                        # yerr=np.abs(10**output_qtl[0] - 10**output_qtl[1]),fmt='o', color = 'green')
            # handles.append(Line2D([0], [0], color='green', marker='o',  label='Predicted value'))
            plt.legend(handles=handles)
            plt.show(block=False)


    return [10**x for x in outputs]



def mass_100_percent_iron_planet(logRadius):
    """
    This is from 100% iron curve of Fortney, Marley and Barnes 2007; solving for logM (base 10) via quadratic formula.
    \nINPUT:
        logRadius : Radius of the planet in log10 units
    OUTPUT:

        logMass: Mass in log10 units for a 100% iron planet of given radius
    """

    Mass_iron = (-0.4938 + np.sqrt(0.4938**2-4*0.0975*(0.7932-10**(logRadius))))/(2*0.0975)
    return Mass_iron

def radius_100_percent_iron_planet(logMass):
    """
    This is from 100% iron curve from Fortney, Marley and Barnes 2007; solving for logR (base 10) via quadratic formula.
    \nINPUT:
        logMass : Mass of the planet in log10 units
    OUTPUT:

        logRadius: Radius in log10 units for a 100% iron planet of given mass
    """

    Radius_iron = np.log10((0.0975*(logMass**2)) + (0.4938*logMass) + 0.7932)
    return Radius_iron

def generate_lookup_table(predict = 'Mass', result_dir = None, cores = 1):
    """
    Generate lookup table size 1000x1000 to make the prediction function faster.
    In log10 units.
    Then in predict_from_measurement() set use_lookup = True.
    \nINPUTS:
        predict_quantity: To predict mass from radius, set to 'mass'. To go the other way,
                          set to 'radius'. Default = 'Mass'
        result_dir: Directory generated by the fitting procedure.
        cores
    OUTPUT:

        The generated lookup table is saved in /result_dir/output/ in the form
        of a .txt file as well as a .npy file which has the 2D interpolated version
        of the lookup table.

    EXAMPLE:

        ## To generate lookup table to get mass from radius
        from mrexo.predict import generate_lookup_table
        kepler_result = '/storage/home/s/szk381/work/mrexo/mrexo/datasets/Kepler_Ning_etal_20170605'
        if __name__ == '__main__':
            generate_lookup_table(result_dir = kepler_result, predict = 'Mass', cores = 10)
    """

    predict_quantity = predict.replace(' ', '').replace('-', '').lower()

    input_location = os.path.join(result_dir, 'input')
    output_location = os.path.join(result_dir, 'output')
    aux_output_location = os.path.join(output_location, 'other_data_products')

    with open(os.path.join(aux_output_location, 'AxesLabels.txt'), 'r') as f:
        LabelDictionary = eval(f.read())

    Y_min, Y_max = np.loadtxt(os.path.join(input_location, 'Y_bounds.txt'))
    X_min, X_max = np.loadtxt(os.path.join(input_location, 'X_bounds.txt'))
    Y_label = LabelDictionary['Y_label'].replace(' ', '').lower()
    X_label = LabelDictionary['X_label'].replace(' ', '').lower()

    lookup_grid_size = 1000

    qtl_steps = np.linspace(0,1,lookup_grid_size)

    if predict_quantity==Y_label:
        search_steps = np.linspace(X_min, X_max, lookup_grid_size)
        fname = 'lookup_y_given_x'
        comment = 'Lookup table for predicting log({}) given log({}) and certain quantile.'.format(Y_label, X_label)
    else:
        search_steps = np.linspace(Y_min, Y_max, lookup_grid_size)
        fname = 'lookup_x_given_y'
        comment = 'Lookup table for predicting log({}) given log({}) and certain quantile.'.format(X_label, Y_label)

    if cores <= 1:
        lookup_table = np.zeros((lookup_grid_size, lookup_grid_size))
        for i in range(0,lookup_grid_size):
            lookup_table[i,:] = np.log10(predict_from_measurement(measurement = 10**search_steps[i], qtl = qtl_steps,
                                result_dir = result_dir, predict = predict_quantity)[1])
            if i%100==0:
                print(i)
    else:
        lookup_inputs = ((10**search_steps[i], qtl_steps, result_dir, predict_quantity) for i in range(lookup_grid_size))
        pool = Pool(processes=cores)
        lookup_table = list(pool.imap(lookup_table_parallelize,lookup_inputs))


    np.savetxt(os.path.join(output_location,fname+'.txt'), lookup_table, comments='#', header=comment)

    interp = interp2d(qtl_steps, search_steps, lookup_table)
    print(interp)
    np.save(os.path.join(output_location,fname+'_interp2d.npy'), interp)

def lookup_table_parallelize(inputs):
    return np.log10(predict_from_measurement(measurement = inputs[0], qtl = inputs[1],
                                result_dir = inputs[2], predict = inputs[3])[1])
