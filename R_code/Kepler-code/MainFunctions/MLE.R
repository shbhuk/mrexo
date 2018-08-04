# By Bo Ning
# Mar. 5. 2018

## pdfnorm.beta: input normal mean and variance, beta shape parameters. 
##               Return the product of normal and beta densities
#' @param x: quantile
#' @param x.obs: observed value
#' @param x.sd: s.d. of a normal distribution
#' @param x.max: upper bound of x
#' @param x.min: lower bound of x
#' @param shape1: the first shape parameter of a beta distribution
#' @param shape2: the second shape parameter of a beta distribution
#' @param log: if x in log scale or not

# gives product of norm and beta densities
pdfnorm.beta <- function(x, x.obs, x.sd, x.max, x.min, shape1, shape2, use_log = TRUE) {
  if (use_log == TRUE) {
    norm.beta <- dnorm(x.obs, mean = 10^x, sd = x.sd) * 
      dbeta((x-x.min)/(x.max-x.min), shape1, shape2)/(x.max - x.min)
  } else {
    norm.beta <- dnorm(x.obs, mean = x, sd = x.sd) * 
      dbeta((x-x.min)/(x.max-x.min), shape1, shape2)/(x.max - x.min)
  }
  return(norm.beta)
}

## fn.for.integrate: integral x from a product of normal and beta density

fn.for.integrate <- function(data, deg, degree, x.max, x.min,
                              use_log = FALSE, abs.tol = 1e-10) {
  integrate(pdfnorm.beta, lower = x.min, upper=x.max, x.obs = data[1], 
            x.sd = data[2], x.max = x.max, x.min = x.min, shape1 = degree,
            shape2 = deg-degree+1, use_log = use_log, abs.tol = abs.tol)$value
}

# calculate marginal densities 
marginal.density <- function(x, x.max, x.min, deg, w.hat) {
  
  x.std <- (x-x.min)/(x.max-x.min)
  
  deg.vec <- 1:deg
  x.beta.indv <- 
    sapply(deg.vec, function(data, degree) {dbeta(data, degree, deg-degree+1)},
           data = x.std) / (x.max-x.min)
  x.beta.pdf <- kronecker(x.beta.indv, rep(1, deg))
  marg.x <- sum(w.hat * x.beta.pdf)
  
  return(marg.x)
}

# calculate conditional densities 
conditional.density <- function(y, y.max, y.min, x, x.max, x.min, deg, w.hat,
                                cond.mean = TRUE, cond.var = TRUE,
                                cond.quantile = TRUE, cond.density = TRUE) {
  deg.vec <- 1:deg
  
  # return conditional mean, variance, quantile, distribution
  y.std <- (y-y.min)/(y.max-y.min)
  y.beta.indv <- sapply(deg.vec, 
                        function(data, degree) {dbeta(data, degree, deg-degree+1)},
                        data = y.std) / (y.max-y.min)
  y.beta.pdf <- kronecker(rep(1, deg), y.beta.indv)
  denominator <- sum(w.hat * y.beta.pdf)
  
  ########### Density ##########
  density.indv.pdf <- sapply(deg.vec, function(x, degree) {
    dbeta(x, degree, deg-degree+1)}, x = (x-x.min)/(x.max-x.min)) / (x.max-x.min)
  density.pdf <- w.hat * kronecker(density.indv.pdf, y.beta.indv) 
  density <- density.pdf/denominator
  
  return(density)
}

#
#x.max = 5
#x.min = 0.01
#x = 0.1
#y = 0.17
#y.max = 11
#y.min = 0.03
#w_hat = 1
#deg = 5 

#y.max = R.max
#y.min = R.min
#x.max = M.max
#x.min = M.min
#deg = deg
#w.hat = w.hat


# calculate 16% and 84% quantiles of a conditional density
cond.density.quantile <- function(y, y.max, y.min, x.max, x.min, deg, w.hat,
                                  qtl = c(0.16, 0.84)) {
  
  deg.vec <- 1:deg 
  
  # return conditional mean, variance, quantile, distribution
  y.std <- (y-y.min)/(y.max-y.min)
  y.beta.indv <- sapply(deg.vec, 
                        function(data, degree) {dbeta(data, degree, deg-degree+1)},
                        data = y.std) / (y.max-y.min)
  y.beta.pdf <- kronecker(rep(1, deg), y.beta.indv)
  denominator <- sum(w.hat * y.beta.pdf)
  
  ########## Mean ############
  mean.beta.indv <- deg.vec/(deg+1)*(x.max-x.min)+x.min
  mean.beta <- kronecker(mean.beta.indv, y.beta.indv)
  mean.nominator <- sum(w.hat * mean.beta)
  mean <- mean.nominator / denominator
  
  ########## Variance ###########
  var.beta.indv <- deg.vec*(deg-deg.vec+1)/((deg+1)^2*(deg+2))*(x.max-x.min)^2
  var.beta <- kronecker(var.beta.indv, y.beta.indv)
  var.nominator <- sum(w.hat * var.beta)
  var <- var.nominator / denominator
  
  ########### Quantile ###########
  # Obtain cdf
  pbeta.conditional.density <- function(x){ 
    
    mix.density <- function(j) {
      
      x.indv.cdf <-
        sapply(deg.vec, 
               function(x, degree) {pbeta(x, degree, deg-degree+1)}, 
               x = (j-x.min)/(x.max-x.min))
      
      quantile.nominator <- sum(w.hat * kronecker(x.indv.cdf, y.beta.indv))
      p.beta <- quantile.nominator/denominator
      return (p.beta)
    }
    
    sapply(x, mix.density)
  }
  
  conditional.quantile <- function(q){ 
    g <- function(x){ pbeta.conditional.density(x)-q }
    return( uniroot(g, interval=c(x.min, x.max) )$root ) 
  }
  quantile <- sapply(qtl, conditional.quantile)
  
  return(c(mean, var, quantile))
}

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
#' @param use_log: is the data transformed into a use_log scale
#' @param abs.tol: precision when calculate the integral 
#' @param output.weights.only only output the estimated weights from the
#'                            Bernstein polynomials if it is TRUE;
#'                            otherwise, output the conditional densities



abs_tol = 1e-20
use_log = TRUE
deg = 5

#MLE.fit(data = data, sigma = sigma, bounds = bounds, deg = deg, use_log = TRUE,abs.tol = 1e-20, output.weights.only = FALSE)


MLE.fit <- function(data, sigma = NULL, bounds, deg, use_log = FALSE,
                    abs.tol = 1e-20, output.weights.only = FALSE) {
  
  library(Rsolnp) # package for solving NLP problem
  
  if (dim(data)[1] < dim(data)[2]) {
    data <- t(data)
  }
  
  # read the data
  n <- dim(data)[1]
  M <- data[, 1]
  R <- data[, 2]
  
  # read the sigma
  if (is.null(sigma) == FALSE) {
    sigma.M <- sigma[, 1]
    sigma.R <- sigma[, 2]
  }
  
  # specify the bounds
  M.max <- bounds[1]
  M.min <- bounds[2]
  R.max <- bounds[3]
  R.min <- bounds[4]
  
  # calculate cdf and pdf of M and R for each term
  deg.vec <- 1:deg
  
  if (is.null(sigma) == TRUE) {
    
    # pdf for Mass for each beta density
    M.indv.pdf <- as.matrix(
      sapply(deg.vec, function(x, degree) {dbeta(x, degree, deg-degree+1)},
             x = (M-M.min)/(M.max-M.min)), n, deg) / (M.max-M.min)
    
    # pdf for Radius for each beta density
    R.indv.pdf <- as.matrix(
      sapply(deg.vec, function(x, degree) {dbeta(x, degree, deg-degree+1)},
             x = (R-R.min)/(R.max-R.min)), n, deg) / (R.max-R.min)
    
    
  } else {

    # pdf for Mass for integrated beta density and normal density
    res.M.pdf <-  
      sapply(deg.vec, 
             function(data, data.sg, deg, degree, x.max, x.min, 
                      use_log = FALSE, abs.tol = 1e-10){
               apply(cbind(data, data.sg), 1, fn.for.integrate,
                     deg = deg, degree = degree, x.max = x.max, x.min = x.min,
                     use_log = use_log, abs.tol = abs.tol)},
             data = M, data.sg = sigma.M, deg = deg, x.max = M.max, x.min = M.min, 
             use_log = use_log, abs.tol = abs.tol)
    M.indv.pdf <- as.matrix(res.M.pdf, n, deg)
    
    
    # pdf for Radius for integrated beta density and normal density
    res.R.pdf <-  
      sapply(deg.vec, 
             function(data, data.sg, deg, degree, x.max, x.min, use_log = FALSE, abs.tol = 1e-10){
               apply(cbind(data, data.sg), 1, fn.for.integrate,
                     deg = deg, degree = degree, x.max = x.max, x.min = x.min,
                     use_log = use_log, abs.tol = abs.tol)},
             data = R, data.sg = sigma.R, deg = deg, x.max = R.max, x.min = R.min,
             use_log = use_log, abs.tol = abs.tol)
    R.indv.pdf <- as.matrix(res.R.pdf, n, deg)
  }
  
  # put M.indv.pdf and R.indv.pdf into a big matrix
  C.pdf <- sapply(1:n, function(x, y, i){kronecker(x[i,], y[i,])}, 
                  x = M.indv.pdf, y = R.indv.pdf)
  
  # Functions use to input solnp solver.
  fn1 <- function(w) {
    if (min(C.pdf) == 0) {
      C.pdf[C.pdf == 0] <- 1e-300
    }
    z <- -sum(log(w %*% C.pdf)) # first and last polynomial is 0
  }
  eqn <- function(w) {
    sum(w)
  }
  
  opt.w <- solnp(rep(1/(deg^2), deg^2), 
                 fun = fn1, eqfun = eqn, eqB = 1, 
                 LB = rep(0, deg^2), UB = rep(1, deg^2), 
                 control=list(trace = 0,outer.iter = 1000))
  if (opt.w$convergence != 0) {
    warning("solnp result does not converge!!!")
  }
  w.hat <- opt.w$pars
  nlog.lik <- min(opt.w$values)

  # calculate aic and bic
  aic <- nlog.lik*2 + 2*(deg^2-1)
  bic <- nlog.lik*2 + log(n) * (deg^2-1)
  
  # marginal densities
  M.seq <- seq(M.min, M.max, length.out = 100)
  R.seq <- seq(R.min, R.max, length.out = 100)
  Mass.marg <- sapply(M.seq, marginal.density, x.max = M.max, x.min = M.min,
                      deg = deg, w.hat = w.hat)
  Radius.marg <- sapply(R.seq, marginal.density, x.max = R.max, x.min = R.min,
                        deg = deg, w.hat = as.vector(t(matrix(w.hat, deg, deg))))
  
  # conditional densities with 16% and 84% quantile
  M.cond.R <- sapply(R.seq[2:99], FUN = cond.density.quantile,
                     y.max = R.max, y.min = R.min,
                     x.max = M.max, x.min = M.min,
                     deg = deg, w.hat = w.hat)
  
  R.cond.M <- sapply(M.seq, FUN = cond.density.quantile,
                     y.max = M.max, y.min = M.min,
                     x.max = R.max, x.min = R.min,
                     deg = deg, w.hat = as.vector(t(matrix(w.hat, deg, deg))))
  M.cond.R.mean <- M.cond.R[1,]
  M.cond.R.var <- M.cond.R[2,]
  M.cond.R.quantile <- M.cond.R[3:4,]
  R.cond.M.mean <- R.cond.M[1,]
  R.cond.M.var <- R.cond.M[2,]
  R.cond.M.quantile <- R.cond.M[3:4,]
  
  
  # return values
  if (output.weights.only == TRUE) {
    return(w.hat)
  } else {
    return(list(weights = w.hat, aic = aic, bic = bic, 
                M.points = M.seq, R.points = R.seq, 
                Mass.marg = Mass.marg, Radius.marg = Radius.marg, 
                M.cond.R = M.cond.R.mean, R.cond.M = R.cond.M.mean,
                M.cond.R.var = M.cond.R.var, R.cond.M.var = R.cond.M.var,
                M.cond.R.quantile = M.cond.R.quantile,
                R.cond.M.quantile = R.cond.M.quantile))
  }
}