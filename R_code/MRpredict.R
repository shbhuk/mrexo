rm(list = ls())
setwd("~/Desktop/MR-relation/MR-predict/") # please change path before running the code

# gives product of norm and beta densities
pdfnorm.beta <- function(x, x.obs, x.sd, x.max, x.min, shape1, shape2, log = TRUE) {
  if (log == TRUE) {
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
                              log = FALSE, abs.tol = 1e-10) {
  integrate(pdfnorm.beta, lower = x.min, upper=x.max, x.obs = data[1], 
            x.sd = data[2], x.max = x.max, x.min = x.min, shape1 = degree,
            shape2 = deg-degree+1, log = log, abs.tol = abs.tol)$value
}

# calculate 16% and 84% quantiles of a conditional density
conditonal.density <- function(y, y.sd = NULL, y.max, y.min, x.max, x.min, 
                                  deg, w.hat, qtl = c(0.16, 0.84)) {
  deg.vec <- 1:deg 
  
  # return conditional mean, variance, quantile, distribution
  if (is.null(y.sd) == TRUE) {
    y.stdardize <- (y-y.min)/(y.max-y.min)
    y.beta.indv <- sapply(deg.vec, 
                          function(data, degree) {dbeta(data, degree, deg-degree+1)},
                          data = y.stdardize) / (y.max-y.min)
  } else {
    y.beta.indv <- sapply(deg.vec, 
                          function(data, degree) 
                          {fn.for.integrate(data, deg, degree, y.max, y.min)},
                          data = c(y, y.sd))
  }
  y.beta.pdf <- kronecker(rep(1, deg), y.beta.indv)
  denominator <- sum(w.hat * y.beta.pdf)
  
  ########## Mean ############
  mean.beta.indv <- deg.vec/(deg+1)*(x.max-x.min)+x.min
  mean.beta <- kronecker(mean.beta.indv, y.beta.indv)
  mean.numerator <- sum(w.hat * mean.beta)
  mean <- mean.numerator / denominator
  
    ########## Variance ###########
    var.beta.indv <- deg.vec*(deg-deg.vec+1)/((deg+1)^2*(deg+2))*(x.max-x.min)^2
    var.beta <- kronecker(var.beta.indv, y.beta.indv)
    var.numerator <- sum(w.hat * var.beta)
    var <- var.numerator / denominator
    
    ########### Quantile ###########
    # Obtain cdf
    pbeta.conditional.density <- function(x){ 
      
      mix.density <- function(j) {
        
        x.indv.cdf <-
          sapply(deg.vec, 
                 function(x, degree) {pbeta(x, degree, deg-degree+1)}, 
                 x = (j-x.min)/(x.max-x.min))
        quantile.numerator <- sum(w.hat * kronecker(x.indv.cdf, y.beta.indv))
        
        p.beta <- quantile.numerator/denominator
        
      }
      
      sapply(x, mix.density)
    }
    
    conditional.quantile <- function(q){ 
      g <- function(x){ pbeta.conditional.density(x)-q }
      return( uniroot(g, interval=c(x.min, x.max) )$root ) 
    }
    
    quantile <- sapply(qtl, conditional.quantile)
    
    return(list(mean = mean, var = var, quantile = quantile,
                denominator = denominator, y.beta.indv))
}


predict.mass.given.radius <- 
  function(Radius, R.sigma = NULL, posterior.sample = FALSE, qtl = c(0.16, 0.84)) {
 
  # upper bounds and lower bounds used in the Bernstein polynomial model in log10 scale
  Radius.min <- -0.3 
  Radius.max <- 1.357509
  Mass.min <- -1
  Mass.max <- 3.809597 
  
  # read weights
  weights.mle <- read.csv("weights.mle.csv")$x 
  
  # convert data into log scale
  l.radius <- log10(Radius)
  
  # The following function can deal with two cases:
  # Case I: if input data do not have measurement errors
  # Case II: if the input data have measurement errors
  if (posterior.sample == FALSE) {
    # convert Radius in log scale
    predicted.value <- 
      conditonal.density(y = l.radius, y.sd = R.sigma, y.max = Radius.max, 
                         y.min = Radius.min, x.max = Mass.max, 
                         x.min = Mass.min, deg = 55, 
                         w.hat = weights.mle, qtl = qtl)
    predicted.mean <- predicted.value$mean
    predicted.lower.quantile <- predicted.value$quantile[1]
    predicted.upper.quantile <- predicted.value$quantile[2]
    
  } else if (posterior.sample == TRUE) {
  
  # Case III: if the input are posterior samples
    
    radius.sample <- log10(Radius)
    deg <- 55
    k <- length(radius.sample) # length of samples
    mean.sample <- denominator.sample <- rep(0, k)
    y.beta.indv.sample <- matrix(0, k, deg)
    
    # the model can be view as a mixture of k conditional densities for f(log m |log r), each has weight 1/k
    # the mean of this mixture density is 1/k times sum of the mean of each conditional density
    # the quantile is little bit hard to compute, and maynot avaible due to computational issues
    
    # calculate the mean
    for (i in 1:k) {
      results <- 
        cond.density.estimation(
          y = radius.sample[i], y.max = Radius.max, 
          y.min = Radius.min, x.max = Mass.max, 
          x.min = Mass.min, deg = 55, 
          w.hat = weights.mle, qtl = quantile,
          only.output.mean = TRUE
        )
      mean.sample[i] <- (results[1])
      denominator.sample[i] <- results[2]
      y.beta.indv.sample[i, ] <- results[3:57]
    }
    predicted.mean <- mean(mean.sample)
    
    # calculate the 16% and 84% quantiles using uniroot function
    # mixture of the CDF of k conditional densities
    pbeta.conditional.density <- function(x){ 
      
      mix.density <- function(j) {
        
        deg.vec <- 1:55
        x.indv.cdf <-
          sapply(deg.vec, 
                 function(x, degree) {pbeta(x, degree, deg-degree+1)}, 
                 x = (j-x.min)/(x.max-x.min))
        quantile.numerator <- rep(0,k)
        p.beta.sample <- rep(0, k)
        for (ii in 1:k) {
          quantile.numerator[ii] <- 
            sum(weights.mle * kronecker(x.indv.cdf, y.beta.indv.sample[ii, ]))
          p.beta.sample[ii] <- quantile.numerator[ii]/denominator.sample[ii]
        }
        p.beta <- mean(p.beta.sample)
      }
      sapply(x, mix.density)
    }
  
    mixture.conditional.quantile <- function(q, x.min, x.max){ 
      g <- function(x){ pbeta.conditional.density(x)-q }
      root <- uniroot(g, interval=c(x.min, x.max) )$root
      return(root) 
    }
    predicted.quantiles <- sapply(qtl, mixture.conditional.quantile,
                                  x.min = Mass.min, x.max = Mass.max)
    predicted.lower.quantile <- predicted.quantiles[1]
    predicted.upper.quantile <- predicted.quantiles[2]
    
  } 
  
  # return
  return(list(predicted.mean = predicted.mean, 
              predicted.lower.quantile = predicted.lower.quantile,
              predicted.upper.quantile = predicted.upper.quantile))
  
}

##################### Examples ######################
# observation without measurement error 
Radius <- 5 # in the original scale, not log scale!!
predict.result <- predict.mass.given.radius(Radius)
print(predict.result) # print out result

# observation with a measurement error
Radius <- 5 # in the original scale, not log scale!!
R.sigma <- 0.1
predict.result <- predict.mass.given.radius(Radius, R.sigma = 0.1)
print(predict.result) # print out result

# input are posterior samples
Radius.samples <- rnorm(100, 5, 0.5)   # in the original scale, not log scale!!
predict.result <- 
  predict.mass.given.radius(Radius = Radius.samples, R.sigma = 0.1, posterior.sample = TRUE)
print(predict.result) # print out result

# if want to change the 16% and 84% quantiles to 5% and 95% quantiles.
Radius <- 5 # in the original scale, not log scale!!
predict.result <- predict.mass.given.radius(Radius, qtl = c(0.05, 0.95))
print(predict.result) # print out result
