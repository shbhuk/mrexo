from mrexo import predict_from_measurement
import os
import numpy as np
pwd = '~/mrexo_working/'

#Below example predicts the mass for a radius of log10(1) Earth radii exoplanet, with no measurement uncertainty from the fit results in 'M_dwarfs_deg_cv'
result_dir = os.path.join(pwd,'M_dwarfs_deg_cv')
predicted_mass, qtls = predict_from_measurement(measurement=1, measurement_sigma=None, result_dir=result_dir, is_posterior=False, is_log=True)

#Below example predicts the mass for a radius of log10(1) Earth radii exoplanet with uncertainty of 0.1 Earth Radii on the included Mdwarf fit. Similary for Kepler dataset.
predicted_mass, qtls = predict_from_measurement(measurement=1, measurement_sigma=0.1, result_dir=None, dataset='mdwarf', is_posterior=False, is_log=True)

#Below example predicts the radius for a mass of log10(1) Earth mass exoplanet with uncertainty of 0.1 Earth Mass on the included Mdwarf fit. Similary for Kepler dataset.
predicted_mass, qtls = predict_from_measurement(measurement=1, measurement_sigma=0.1, predict = 'radius', result_dir=None, dataset='mdwarf', is_posterior=False, is_log=True)    


def find_mass_probability_distribution_function(R_check, Radius_min, Radius_max, Mass_max, Mass_min, weights_mle, weights_boot, degree, deg_vec, M_points, is_log = True):
    '''

    '''

    if is_log == False:
        R_check = np.log10(R_check)

    n_quantiles = 200
    qtl = np.linspace(0,1.0, n_quantiles)

    results = cond_density_quantile(y=R_check, y_std=None, y_max=Radius_max, y_min=Radius_min,
                                                      x_max=Mass_max, x_min=Mass_min, deg=degree, deg_vec=deg_vec,
                                                      w_hat=weights_mle, qtl=qtl)

    cdf_interp = interp1d(results[2], qtl)(M_points)

    # Conditional_plot. PDF is derivative of CDF
    pdf_interp = np.diff(cdf_interp) / np.diff(M_points)

    n_boot = np.shape(weights_boot)[0]
    pdf_boots = np.zeros((n_boot, len(M_points) - 1))
    cdf_interp_boot = np.zeros((n_boot, len(M_points)))

    for i in range(0, n_boot):
        weight = weights_boot[i,:]
        results_boot = cond_density_quantile(y=R_check, y_std=None, y_max=Radius_max, y_min=Radius_min,
                                                        x_max=Mass_max, x_min=Mass_min, deg=degree, deg_vec=deg_vec,
                                                        w_hat=weight, qtl=qtl)
        cdf_interp_boot[i,:] = interp1d(results_boot[2], qtl)(M_points)
        pdf_boots[i,:] = np.diff(cdf_interp_boot[i,:]) / np.diff(M_points)
        print(i)

    lower_boot, upper_boot = mquantiles(pdf_boots ,prob=[0.16, 0.84],axis=0,alphap=1,betap=1).data

    return cdf_interp, pdf_interp, cdf_interp_boot, lower_boot, upper_boot


def find_radius_probability_distribution_function(M_check, Mass_max, Mass_min, Radius_min, Radius_max, weights_mle, weights_boot, degree, deg_vec, R_points, is_log = True):
    '''

    '''

    if is_log == False:
        M_check = np.log10(M_check)

    n_quantiles = 200
    qtl = np.linspace(0,1.0, n_quantiles)

    results = cond_density_quantile(y=M_check, y_std=None, y_max=Mass_max, y_min=Mass_min,
                                                      x_max=Radius_max, x_min=Radius_min, deg=degree, deg_vec=deg_vec,
                                                      w_hat=weights_mle, qtl=qtl)

    cdf_interp = interp1d(results[2], qtl)(R_points)

    # Conditional_plot. PDF is derivative of CDF
    pdf_interp = np.diff(cdf_interp) / np.diff(R_points)

    n_boot = np.shape(weights_boot)[0]

    cdf_interp_boot = np.zeros((n_boot, len(R_points)))
    pdf_boots = np.zeros((n_boot, len(R_points) - 1))

    for i in range(0, n_boot):
        weight = weights_boot[i,:]
        results_boot = cond_density_quantile(y=M_check, y_std=None, y_max=Mass_max, y_min=Mass_min,
                                                        x_max=Radius_max, x_min=Radius_min, deg=degree, deg_vec=deg_vec,
                                                        w_hat=weight, qtl=qtl)
        cdf_interp_boot[i,:] = interp1d(results_boot[2], qtl)(R_points)
        pdf_boots[i,:] = np.diff(cdf_interp_boot[i,:]) / np.diff(R_points)
        print(i)

    lower_boot, upper_boot = mquantiles(pdf_boots ,prob=[0.16, 0.84],axis=0,alphap=1,betap=1).data

    return cdf_interp, pdf_interp, cdf_interp_boot, lower_boot, upper_boot
