# By Bo Ning
# Mar. 5. 2018

# Cross validation function
#' @param data.train: input training data
#' @param data.sg.train: input measurement errors for the training data
#' @param bounds: a vector contains four elements, from left to right:
#'                the upper bound for mass, the lower bound for mass
#'                the upper bound for radius, the lower bound for radius
#' @param data.test: input test data
#' @param data.sg.test: input measurement errors for the test data
#' @param deg: degree used in Bernstein polynomials
#' @param abs.tol: tolerance number of computing

cross.validation <- function(data.train, data.sg.train = NULL, bounds, 
                             data.test, data.sg.test = NULL, 
                             deg, log = FALSE, abs.tol = 1e-20) {
  
  # fit MLE using training dataset
  weights <- MLE.fit(data = data.train, sigma = data.sg.train, 
                     bounds = bounds, deg = deg, log = log,
                     abs.tol = abs.tol, output.weights.only = TRUE)
  
  # calculate log-likelihood using test datasets based on estimated weights
  # using training data
  if (dim(as.matrix(data.test))[2] == 1) {
    n.test <- 1
    M.test <- data.test[1]
    R.test <- data.test[2]
  } else {
    n.test <- dim(data.test)[1]
    M.test <- data.test[, 1]
    R.test <- data.test[, 2]
  }
  
  # read the sigma
  if (is.null(data.sg.test) == FALSE) {
    if (dim(as.matrix(data.sg.test))[2] == 1) {
      M.sg.test <- data.sg.test[1]
      R.sg.test <- data.sg.test[2]
    } else {
      M.sg.test <- data.sg.test[, 1]
      R.sg.test <- data.sg.test[, 2]
    }
  }
    
  # specify the bounds
  M.max <- bounds[1]
  M.min <- bounds[2]
  R.max <- bounds[3]
  R.min <- bounds[4]
  
  # calculate cdf and pdf of M and R for each term
  # the first and last term is set to 0 to avoid boundary effects
  # so we only need to calculate 2:(deg^2-1) terms
  deg.vec <- 1:deg
  
  if (is.null(data.sg.test) == TRUE) {
    
    # pdf for Mass for each beta density
    M.indv.pdf <- as.matrix(
      sapply(deg.vec, function(x, degree) {dbeta(x, degree, deg-degree+1)},
             x = (M.test-M.min)/(M.max-M.min)), n.test, deg) / (M.max-M.min)
    
    # pdf for Radius for each beta density
    R.indv.pdf <- as.matrix(
      sapply(deg.vec, function(x, degree) {dbeta(x, degree, deg-degree+1)},
             x = (R.test-R.min)/(R.max-R.min)), n.test, deg) / (R.max-R.min)
    
    
  } else {
    
    # pdf for Mass for integrated beta density and normal density
    res.M.pdf <-  
      sapply(deg.vec, 
             function(data, data.sg, deg, degree, x.max, x.min, 
                      log = FALSE, abs.tol = 1e-10){
               apply(cbind(data, data.sg), 1, fun.for.integrate,
                     deg = deg, degree = degree, x.max = x.max, x.min = x.min,
                     log = log, abs.tol = abs.tol)},
             data = M.test, data.sg = M.sg.test, deg = deg, x.max = M.max, x.min = M.min, 
             log = log, abs.tol = abs.tol)
    M.indv.pdf <- as.matrix(res.M.pdf, n.test, deg)
    
    # pdf for Radius for integrated beta density and normal density
    res.R.pdf <-  
      sapply(deg.vec, 
             function(data, data.sg, deg, degree, x.max, x.min, 
                      log = FALSE, abs.tol = 1e-10){
               apply(cbind(data, data.sg), 1, fun.for.integrate,
                     deg = deg, degree = degree, x.max = x.max, x.min = x.min,
                     log = log, abs.tol = abs.tol)},
             data = R.test, data.sg = R.sg.test, deg = deg, x.max = R.max, x.min = R.min,
             log = log, abs.tol = abs.tol)
    R.indv.pdf <- as.matrix(res.R.pdf, n.test, deg)
  }
  
  if (n.test == 1) {
    C.pdf <- kronecker(M.indv.pdf, R.indv.pdf)
  } else {
    C.pdf <- sapply(1:n.test, function(x, y, i){kronecker(x[i,], y[i,])}, 
                    x = M.indv.pdf, y = R.indv.pdf)
  }
  
  loglike.pred <- sum(log(weights %*% C.pdf))
  
  return(loglike.pred)
}


