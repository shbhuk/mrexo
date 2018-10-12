# By Bo Ning
# Mar. 5. 2018

## MRpredict: predict the Mass and Radius relationship
#' @param data: the first column contains the mass measurements and 
#'              the second column contains the radius measurements.
#' @param sigma: measurement errors for the data, if no measuremnet error, 
#'               it is NULL
#' @param Mass.max: the upper bound for mass
#' @param Mass.min: the lower bound for mass
#' @param Radius.max: the upper bound for radius
#' @param Radius.min: the upper bound for radius
#' @param degree: the maximum degree used for cross-validation/AIC/BIC
#' @param selected.deg: if input "cv": cross validation
#'                      if input "aic": aic method
#'                      if input "bic": bic method
#'                      if input a number: default using that number and 
#'                      skip the select process 
#' @param log: is the data transformed into a log scale
#' @param k.fold: number of fold used for cross validation, default is 10
#' @param bootstrap: if using bootstrap to obtain confidence interval, 
#'                   input TRUE
#' @param num.boot: number of bootstrap replication
#' @param store.output: store the output into csv files if TRUE
#' @param cores: this program uses parallel computing for bootstrap,
#'               default cores are 7

MRpredict <- function(data, sigma, Mass.min = NULL, Mass.max = NULL,
                      Radius.min = NULL, Radius.max = NULL, degree = 60, 
                      log = FALSE, select.deg = 55, k.fold = 10,
                      bootstrap = TRUE, num.boot = 100, store.output = FALSE,
                      cores = 7) {
  
  # check if a package has been installed
  pkgTest <- function(x)
  {
    if (!require(x,character.only = TRUE)) {
      install.packages(x,dep=TRUE)
      if(!require(x,character.only = TRUE)) stop("Package not found")
    } 
  }
  pkgTest("Rsolnp")
  pkgTest("parallel")
  
  # load function
  library(Rsolnp)
  
  ###########################################################
  Mass.obs <- data[, 1]  # mass
  Radius.obs <- data[, 2] # radius
  Mass.sigma <- sigma[, 1] # measurement errors for masses
  Radius.sigma <- sigma[, 2] # measurement errors for radii
  
  ## Step 0: organize the dataset.
  n <- length(Mass.obs) # num of obs.
  if (length(Mass.obs) != length(Radius.obs)) {
    warnings("The length of Mass and Radius must be the same!!!")
  }
  
  # rename the variables
  M.obs <- Mass.obs
  R.obs <- Radius.obs
  
  if (is.null(Mass.sigma) == FALSE) {
    if (length(Mass.sigma != n)) {
      warnings("The length of Mass and Mass.sigma must be the same!!!")
    }
    M.sg <- Mass.sigma
  } else {M.sg <- rep(0, n)}
  
  if (is.null(Radius.sigma) == FALSE) {
    if (length(Radius.sigma != n)) {
      warnings("The length of Mass and Mass.sigma must be the same!!!")
    }
    R.sg <- Radius.sigma
  } else {R.sg <- rep(0, n)}
  
  if (is.null(Mass.min) == TRUE) {
    M.min <- min(M.obs) - max(M.sg)/sqrt(n)
  } else {M.min <- Mass.min}
  
  if (is.null(Mass.max) == TRUE) {
    M.max <- max(M.obs) + max(M.sg)/sqrt(n)
  } else {M.max <- Mass.max}
  
  if (is.null(Radius.min) == TRUE) {
    R.min <- min(R.obs) - max(R.sg)/sqrt(n)
  } else {R.min <- Radius.min}
  
  if (is.null(Radius.max) == TRUE) {
    R.max <- max(R.obs) + max(R.sg)/sqrt(n)
  } else {R.max <- Radius.max}
  
  bounds <- c(M.max, M.min, R.max, R.min)
  
  # load functions
  source("MainFunctions/MLE.R")
  source("MainFunctions/cross-validation.R")
  
  # organize dataset
  data <- cbind(M.obs, R.obs) 
  if (is.null(Mass.sigma) == FALSE) {
    data.sg <- cbind(M.sg, R.sg)
  }
  
  ###########################################################
  ## Step 1: Select number of degree based on cross validation, aic or bic methods.
  degree <- degree
  k.fold <- k.fold
  
  degree.candidate <- seq(5, degree, by = 5)
  deg.length <- length(degree.candidate)
  
  if (select.deg == "cv") {
    
    library(parallel)
    
    k.fold <- k.fold # number of fold for cross validation, default is 10
    rand.gen <- sample(1:n, n) # randomly shuffle the dataset
    lik.per.degree <- rep(NA, deg.length)
    
    cv.parallel.fn <- function(x) {
      
      i.fold <- x
      if (i.fold < k.fold) {
        split.interval <- ((i.fold-1)*floor(n/k.fold)+1):(i.fold*floor(n/k.fold))
      } else {
        split.interval <- ((i.fold-1)*floor(n/k.fold)+1):n
      }
      
      data.train <- data[ -rand.gen[split.interval], ]
      data.test <- data[ rand.gen[split.interval], ]
      data.sg.train <- data.sg[ -rand.gen[split.interval], ]
      data.sg.test <- data.sg[ rand.gen[split.interval], ]
      like.pred <- 
        cross.validation(data.train, data.sg.train, bounds, 
                         data.test, data.sg.test, degree.candidate[i.degree],
                         log = FALSE)
      return(like.pred)
    }
    
    for (i.degree in 1:deg.length) {

      like.pred.vec <- rep(NA, k.fold)
      for (i.fold in 1:k.fold) {

        # creat indicator to separate dataset into training and testing datasets
        if (i.fold < k.fold) {
          split.interval <- ((i.fold-1)*floor(n/k.fold)+1):(i.fold*floor(n/k.fold))
        } else {
          split.interval <- ((i.fold-1)*floor(n/k.fold)+1):n
        }

        data.train <- data[ -rand.gen[split.interval], ]
        data.test <- data[ rand.gen[split.interval], ]
        data.sg.train <- data.sg[ -rand.gen[split.interval], ]
        data.sg.test <- data.sg[ rand.gen[split.interval], ]
        like.pred <-
          cross.validation(data.train, data.sg.train, bounds,
                           data.test, data.sg.test, degree.candidate[i.degree])
        like.pred.vec[i.fold] <- like.pred

      }
      lik.per.degree[i.degree] <- sum(like.pred.vec)
      cat("deg = ", degree.candidate[i.degree], "like.cv = ", lik.per.degree[i.degree], "\n")
    }
    
  } else if (select.deg == "aic") {
    
    aic <- rep(NA, deg - 1)
    for (d in 2:deg) {
      MR.fit <- MLE.fit(data, bounds, data.sg, deg = d, output.density = F)
      aic[d-1] <- MR.fit$aic
    }
    
    deg.choose <- which(aic == min(aic)) 
  } else if (select.deg == "bic") {
    bic <- rep(NA, deg - 1)
    for (d in 2:deg) {
      MR.fit <- MLE.fit(data, bounds, data.sg, deg = d, output.density = F)
      bic[d-1] <- MR.fit$bic
    }
    
    deg.choose <- which(bic == min(bic)) 
  } else {
    deg.choose <- select.deg
  }
  
  ###########################################################
  ## Step 2: Estimate the model
  MR.MLE <- MLE.fit(data = data, bounds = bounds, sigma = data.sg, 
                    deg = deg.choose, log = TRUE)
  
  if (bootstrap == TRUE) {
    
    # weights <- Mass.marg.boot <- Radius.marg.boot <- 
    #   M.cond.R.boot.var <- M.cond.R.boot <- R.cond.M.boot <- list() 
    pb  <- txtProgressBar(1, num.boot, style=3)
    cat("\nStarting Bootstrap: \n")
    
    boot.parallel.fn <- function(rep) {
      # setTxtProgressBar(pb, rep)
      print(rep) 
      
      n.boot <- sample(1:n, replace = T)
      data.boot <- data[n.boot, ]
      data.sg.boot <- data.sg[n.boot, ]
      MR.boot <- MLE.fit(data.boot, data.sg.boot, bounds, deg = deg.choose, log = log)
    }
    
    library(parallel)
    result <- mclapply(1:num.boot, boot.parallel.fn, mc.cores = cores)
    
    weights.boot <- Mass.marg.boot <- Radius.marg.boot <- 
      M.cond.R.boot <- M.cond.R.var.boot <- R.cond.M.boot <- R.cond.M.var.boot <- 
      M.cond.R.quantile.boot <- R.cond.M.quantile.boot <- NULL
    
    M.points <- result[[1]]$M.points
    R.points <- result[[1]]$R.points
    
    for (i.boot in 1:num.boot) {
      weights.boot <- cbind(weights.boot, result[[i.boot]]$weights)
      Mass.marg.boot <- cbind(Mass.marg.boot, result[[i.boot]]$Mass.marg)
      Radius.marg.boot <- cbind(Radius.marg.boot, result[[i.boot]]$Radius.marg)
      M.cond.R.boot <- cbind(M.cond.R.boot, result[[i.boot]]$M.cond.R)
      M.cond.R.var.boot <- cbind(M.cond.R.var.boot, result[[i.boot]]$M.cond.R.var)
      M.cond.R.quantile.boot <- cbind(M.cond.R.quantile.boot, 
                                      t(result[[i.boot]]$M.cond.R.quantile))
      R.cond.M.boot <- cbind(R.cond.M.boot, result[[i.boot]]$R.cond.M)
      R.cond.M.var.boot <- cbind(R.cond.M.var.boot, result[[i.boot]]$R.cond.M.var)
      R.cond.M.quantile.boot <- cbind(R.cond.M.quantile.boot, 
                                      t(result[[i.boot]]$R.cond.M.quantile))
    }
    
    
    if (store.output == TRUE) {
      write.csv(M.points, "M.points.Rdata")
      write.csv(R.points, "R.points.Rdata")
      write.csv(Mass.marg.boot, "Mass.marg.boot.Rdata")
      write.csv(Radius.marg.boot, "Radius.marg.boot.Rdata")
      write.csv(M.cond.R.boot, "M.cond.R.boot.Rdata")
      write.csv(R.cond.M.boot, "R.cond.M.boot.Rdata")
      write.csv(M.cond.R.var.boot, "M.cond.R.var.boot.Rdata")
      write.csv(M.cond.R.lower.boot, "M.cond.R.lower.boot.Rdata")
      write.csv(M.cond.R.upper.boot, "M.cond.R.upper.boot.Rdata")
      write.csv(R.cond.M.var.boot, "R.cond.M.var.boot.Rdata")
      write.csv(R.cond.M.lower.boot, "R.cond.M.lower.boot.Rdata")
      write.csv(R.cond.M.upper.boot, "R.cond.M.upper.boot.Rdata")
    }
  }
  
  return(list(Mass.marg.boot = Mass.marg.boot, Radius.marg.boot = Radius.marg.boot,
              M.cond.R.boot = M.cond.R.boot, R.cond.M.boot = R.cond.M.boot,
              M.cond.R.var.boot = M.cond.R.var.boot))
}