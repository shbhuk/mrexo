# MRExo 
Non parametric exoplanet mass radius relationship

We translate Ning et al. (2018)’s `R` script\[1\] into a publicly available `Python`
package called `MRExo`\[2\]. 

`MRExo` offers tools for fitting the M-R relationship to a given data
set. In this package we use a cross validation technique which randomly
samples the data to optimize for the number of degrees. We then fit the
joint distribution for the
sample set; this can then be marginalized to obtain the conditional
distribution, which we can use to predict one variable from the other.
Further, `MRExo` is equipped with dedicated and easy to use functions to
plot the best fitted conditional and joint M-R relationships, and also
to predict mass from radius, and radius from mass. For example, in the
case of planets discovered using the transit method, the feasibility of
an RV follow-up campaign can be evaluated by predicting the estimated
mass and its confidence intervals given the radius and uncertainty
measured in transit. Another feature of this package is that, it can be
used to predict posteriors for masses (or radii) given posteriors from
radii (or masses); it also outputs the marginal and conditional
distributions. We bootstrap our fitting procedure to estimate the
uncertainties to the mean. Along with the `MRExo` installation, the fit
results from the M dwarf sample dataset from Kanodia et al. (2019) and the Kepler
exoplanet sample from  Ning et al. (2018) are included.

The number of degrees for the Bernstein polynomials approximately scales
with the sample size. Since the number of weights goes as degree squared
the computation time involved in the fitting a new M-R can soon start to
become prohibitive. Therefore we also parallelize the fitting procedure
and the bootstrapping algorithm. As an example, for the M dwarf sample
size of 24 points which required 17 degrees took about 8 minutes to
cross validate for degrees, fit a relationship, and do 100 bootstraps on
a cluster node with 24 cores and 2.2 GHz processors. Our simulation
study of 100 points and 55 degrees took about 2 days for the cross
validation, fitting, and 24 bootstraps. We realize that the fitting
computation time would start to become prohibitive as the sample set
increases (~ 200), we therefore plan to optimize the code further
by benchmarking, floating point operation optimization, and correcting
the precision requirements in the integration step. However, this time
intensive step of cross validation and fitting is only if the user needs
to fit their own relationship. To run the prediction routine on the
pre-existing M dwarf or Kepler sample is quick. In order to do a large
number of predictions as part of a larger pipeline or simulation, the user can also generate a look-up table which makes the calculations even faster (the function to generate and use the look-up table is provided with the package)


1.  <https://github.com/Bo-Ning/Predicting-exoplanet-mass-and-radius-relationship>.

2.  <https://github.com/shbhuk/mrexo>
