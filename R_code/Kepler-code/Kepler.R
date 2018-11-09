#setwd('C:/Users/szk381/Documents/GitHub/Predicting-exoplanet-mass-and-radius-relationship/Kepler-code/Result-Kepler/')
setwd('C:/Users/shbhu/Documents/GitHub/Py_mass_radius_working/R_code/Kepler-code/')
##### Read the dataset #######
rm(list = ls())
raw.data <- read.csv(file = "MR_Kepler_170605_noanalytTTV_noupplim.csv", skip = 49)
data <- subset(raw.data, select = c("rowid", "pl_hostname", "pl_masse",
                                    "pl_masseerr1", "pl_masseerr2",
                                    "pl_rade", "pl_radeerr1", "pl_radeerr2"))

# taking sigma of Mass and Radius into half
Mass.sigma <- (data$pl_masseerr1 + abs(data$pl_masseerr2))/2
Radius.sigma <- (data$pl_radeerr1 + abs(data$pl_radeerr2))/2

Mass.obs <- raw.data$pl_masse
Radius.obs <- raw.data$pl_rade
# bounds for Mass and Radius
Radius.min <- -0.3
Radius.max <- log10(max(Radius.obs) + sd(Radius.obs)/sqrt(length(Radius.obs)))
Mass.min <- log10( max(min(Mass.obs) - sd(Mass.obs)/sqrt(length(Mass.obs)), 0.1))
Mass.max <- log10(max(Mass.obs) + sd(Mass.obs)/sqrt(length(Mass.obs)))
num.boot <- 100
select.deg <- 55

###########################################################
## Step 0: organize the dataset.
n <- length(Mass.obs) # num of obs.
if (length(Mass.obs) != length(Radius.obs)) {
  warnings("The length of Mass and Radius must be the same!!!")
}

# rename the variables
data <- cbind(Mass.obs, Radius.obs)
sigma <- cbind(Mass.sigma, Radius.sigma)
M.obs <- Mass.obs
R.obs <- Radius.obs
M.sg <- Mass.sigma
R.sg <- Radius.sigma
bounds <- c(Mass.max, Mass.min, Radius.max, Radius.min)

source("MainFunctions/MRpredict.R")

result <- MRpredict(data, sigma, Mass.min = Mass.min, Mass.max = Mass.max,
                    Radius.min = Radius.min, Radius.max = Radius.max, 
                    log = TRUE, select.deg = 2, 
                    bootstrap = TRUE, num.boot = 1, store.output = FALSE,
                    cores = 1)

M.points <- result$M.points
R.points <- result$R.points
Mass.marg.boot <- result$Mass.marg.boot
Radius.marg.boot <- result$Radius.marg.boot
M.cond.R.boot <- result$M.cond.R.boot
R.cond.M.boot <- result$R.cond.M.boot
M.cond.R.var.boot <- result$M.cond.R.var.boot
R.cond.M.var.boot <- result$R.cond.M.var.boot
M.cond.R.lower.boot <- result$M.cond.R.lower.boot
R.cond.M.lower.boot <- result$R.cond.M.lower.boot
M.cond.R.upper.boot <- result$M.cond.R.upper.boot
R.cond.M.upper.boot <- result$R.cond.M.upper.boot

write.csv(M.points, "M.points.csv")
write.csv(R.points, "R.points.csv")
write.csv(Mass.marg.boot, "Mass.marg.boot.csv")
write.csv(Radius.marg.boot, "Radius.marg.boot.csv")
write.csv(M.cond.R.boot, "M.cond.R.boot.csv")
write.csv(R.cond.M.boot, "R.cond.M.boot.csv")
write.csv(M.cond.R.var.boot, "M.cond.R.var.boot.csv")
write.csv(M.cond.R.lower.boot, "M.cond.R.lower.boot.csv")
write.csv(M.cond.R.upper.boot, "M.cond.R.upper.boot.csv")
write.csv(R.cond.M.var.boot, "R.cond.M.var.boot.csv")
write.csv(R.cond.M.lower.boot, "R.cond.M.lower.boot.csv")
write.csv(R.cond.M.upper.boot, "R.cond.M.upper.boot.csv")
