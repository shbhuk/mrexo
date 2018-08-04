rm(list = ls())
setwd("~/Documents/Research/[2017Astro]NonparamM-R/code/Kepler-result/")
# setwd("~/Desktop/Kepler/")

raw.data <- read.csv(file = "MR_Kepler_170605_noanalytTTV_noupplim.csv", skip = 49)
data <- subset(raw.data, select = c("rowid", "pl_hostname", "pl_masse",
                                    "pl_masseerr1", "pl_masseerr2",
                                    "pl_rade", "pl_radeerr1", "pl_radeerr2"))

# taking sigma of Mass and Radius into half
Mass.sigma <- (data$pl_masseerr1 + abs(data$pl_masseerr2))/2
Radius.sigma <- (data$pl_radeerr1 + abs(data$pl_radeerr2))/2
Mass.obs <- raw.data$pl_masse
Radius.obs <- raw.data$pl_rade
Radius.min <- -0.3
Radius.max <- log10(max(Radius.obs) + sd(Radius.obs)/sqrt(length(Radius.obs)))
Mass.min <- log10( max(min(Mass.obs) - sd(Mass.obs)/sqrt(length(Mass.obs)), 0.1))
Mass.max <- log10(max(Mass.obs) + sd(Mass.obs)/sqrt(length(Mass.obs)))

M.points <- read.csv("M.points.csv")$x
R.points <- read.csv("R.points.csv")$x
weights.mle <- read.csv("weights.mle.csv")$x
Mass.marg <- read.table("Mass.marg.Rdata")
Radius.marg <- read.table("Radius.marg.Rdata")
M.cond.R <- read.table("M.cond.R.Rdata")
R.cond.M <- read.table("R.cond.M.Rdata")
M.cond.R.var <- read.table("M.cond.R.var.Rdata")
M.cond.R.lower <- read.table("M.cond.R.lower.Rdata")
M.cond.R.upper <- read.table("M.cond.R.upper.Rdata")
R.cond.M.var <- read.table("R.cond.M.var.Rdata")
R.cond.M.lower <- read.table("R.cond.M.lower.Rdata")
R.cond.M.upper <- read.table("R.cond.M.upper.Rdata")

weights.boot <- read.csv("weights.boot.csv")
Mass.marg.boot <- read.csv("Mass.marg.boot.csv")
Radius.marg.boot <- read.csv("Radius.marg.boot.csv")
M.cond.R.boot <- read.csv("M.cond.R.boot.csv")
R.cond.M.boot <- read.csv("R.cond.M.boot.csv")
M.cond.R.var.boot <- read.csv("M.cond.R.var.boot.csv")
M.cond.R.quantile.boot <- read.csv("M.cond.R.quantile.boot.csv")
R.cond.M.var.boot <- read.csv("R.cond.M.var.boot.csv")
R.cond.M.quantile.boot <- read.csv("R.cond.M.quantile.boot.csv")

Mass.marg <- as.numeric(gsub(",", "", as.character(Mass.marg$V2)))[2:101]
Radius.marg <- as.numeric(gsub(",", "", as.character(Radius.marg$V2)))[2:101]
M.cond.R <- as.numeric(gsub(",", "", as.character(M.cond.R$V2)))[2:101]
R.cond.M <- as.numeric(gsub(",", "", as.character(R.cond.M$V2)))[2:101]
M.cond.R.var <- as.numeric(gsub(",", "", as.character(M.cond.R.var$V2)))[2:101]
M.cond.R.lower <- as.numeric(gsub(",", "", as.character(M.cond.R.lower$V2)))[2:101]
M.cond.R.upper <- as.numeric(gsub(",", "", as.character(M.cond.R.upper$V2)))[2:101]
R.cond.M.var <- as.numeric(gsub(",", "", as.character(R.cond.M.var$V2)))[2:101]
R.cond.M.lower <- as.numeric(gsub(",", "", as.character(R.cond.M.lower$V2)))[2:101]
R.cond.M.upper <- as.numeric(gsub(",", "", as.character(R.cond.M.upper$V2)))[2:101]

weights.boot <- as.matrix(weights.boot)[, 2:101]
Mass.marg.boot <- as.matrix(Mass.marg.boot)[, 2:101]
Radius.marg.boot <- as.matrix(Radius.marg.boot)[, 2:101]
M.cond.R.boot <- as.matrix(M.cond.R.boot)[, 2:101]
R.cond.M.boot <- as.matrix(R.cond.M.boot)[, 2:101]
M.cond.R.var.boot <- as.matrix(M.cond.R.var.boot)[, 2:101]
M.cond.R.quantile.boot <- as.matrix(M.cond.R.quantile.boot)[, 2:201]
M.cond.R.lower.boot <- M.cond.R.quantile.boot[, seq(1, 200, by = 2)]
M.cond.R.upper.boot <- M.cond.R.quantile.boot[, seq(2, 200, by = 2)]
R.cond.M.var.boot <- as.matrix(R.cond.M.var.boot)[, 2:101]
R.cond.M.quantile.boot <- as.matrix(R.cond.M.quantile.boot)[, 2:201]
R.cond.M.lower.boot <- R.cond.M.quantile.boot[, seq(1, 200, by = 2)]
R.cond.M.upper.boot <- R.cond.M.quantile.boot[, seq(2, 200, by = 2)]


# save-to-pdf function
savepdf <- function(pdf.name, Myplot) {
  pdf(file = pdf.name, width = 6, height = 4.5)
  print(Myplot)
  dev.off()
}

##################### Figure 4 #######################
# orgnize dataset
nonparam.pred.df <- data.frame(
  radius = R.points,
  mean = M.cond.R, 
  upper = M.cond.R.upper,
  lower = M.cond.R.lower
)

library(matrixStats)

nonparam.boot.df <- data.frame(
  radius = R.points,
  upper.boot = rowQuantiles(M.cond.R.boot, probs = 0.84),
  lower.boot = rowQuantiles(M.cond.R.boot, probs = 0.16)
)

# calculate log-normal quantile
lM.obs <- log10(Mass.obs)
lR.obs <- log10(Radius.obs)
lM.sd.obs <- 0.434*Mass.sigma/Mass.obs
lR.sd.obs <- 0.434*Radius.sigma/Radius.obs
data.points <- data.frame(lnM.obs = lM.obs, lnR.obs = lR.obs,
                          lnM.max = lM.obs+lM.sd.obs,
                          lnM.min = lM.obs-lM.sd.obs,
                          lnR.max = lR.obs+lR.sd.obs,
                          lnR.min = lR.obs-lR.sd.obs)

darkblue <- rgb(0,49,87,maxColorValue = 256)
lightblue <- rgb(5,122,255,maxColorValue = 256)
yellow <- rgb(242,208,58,maxColorValue = 256)


library(ggplot2)
library(extrafont)
figureData <-  ggplot() +
  geom_point(data = data.points, aes(x = lnR.obs, y = lnM.obs), 
             col = "black", alpha = 1, size = 1) +
  geom_errorbar(data = data.points, 
                aes(x = lnR.obs, ymin = lnM.min, ymax = lnM.max),
                col = "grey20", alpha = 1, size = 0.4) +
  geom_errorbarh(data = data.points, 
                 aes(x = lnR.obs, y = lnM.obs, xmin = lnR.min, xmax = lnR.max),
                 col = "grey20", alpha = 1, size = 0.4) +
  xlab(expression(paste("Radius (R"["Earth"],")"))) +
  ylab(expression(paste("Mass (M"["Earth"],")"))) + 
  coord_cartesian(ylim = c(-1.5,4), xlim = c(-0.3, 1.5)) +
  scale_y_continuous(breaks=c(-1, 0, 1, 2, 3, 4),
                     labels=c(expression(bold("0.1")),
                              expression(bold("1")),
                              expression(bold("10")),
                              expression(bold(paste("10"^"2"))),
                              expression(bold(paste("10"^"3"))),
                              expression(bold(paste("10"^"4"))))) +
  scale_x_continuous(breaks=c(-0.2, log10(1),log10(2),log10(3), log10(4),
                              log10(5),log10(7),log10(10),log10(15),
                              log10(20),log10(30)),
                     labels=c(0.6, 1,2,3,4,5,7,10,15,20,30)) + 
  theme(axis.text=element_text(size=15),
        axis.title=element_text(size=15,face="bold"),
        plot.title = element_text(hjust = 0.5),
        text=element_text(family="Helvetica", size=12, face = "bold"),
        panel.background = element_blank(),
        axis.line = element_line(colour = "black")) +
  ggtitle("Kepler data") 
pdf.name <- "figureData.pdf"
savepdf <- function(pdf.name, Myplot) {
  pdf(file = pdf.name, width = 6, height = 4)
  print(Myplot)
  dev.off()
}
savepdf(pdf.name, figureData)



figure4 <-  ggplot() +
  geom_point(data = data.points, aes(x = lnR.obs, y = lnM.obs), 
             col = "grey10", alpha = 0.6, size = 0.5) +
  geom_errorbar(data = data.points, 
                aes(x = lnR.obs, ymin = lnM.min, ymax = lnM.max),
                col = "grey20", alpha = 0.6, size = 0.15) +
  geom_errorbarh(data = data.points, 
                 aes(x = lnR.obs, y = lnM.obs, xmin = lnR.min, xmax = lnR.max),
                 col = "grey20", alpha = 0.6, size = 0.15) +
  geom_path(data = nonparam.pred.df, aes(radius, mean),
            color = darkblue, size = 1) +
  geom_ribbon(data = nonparam.pred.df, 
              aes(x = radius, ymin = lower, ymax = upper),
              fill = darkblue, alpha = 0.3) +
  geom_ribbon(data = nonparam.boot.df, 
              aes(x = radius, ymin = lower.boot, ymax = upper.boot),
              fill = lightblue, alpha = 0.3) +
  xlab(expression(paste("Radius (R"["Earth"],")"))) +
  ylab(expression(paste("Mass (M"["Earth"],")"))) + 
  coord_cartesian(ylim = c(-1.5,4), xlim = c(-0.3, 1.5)) +
  scale_y_continuous(breaks=c(-1, 0, 1, 2, 3, 4),
                     labels=c(expression(bold("0.1")),
                              expression(bold("1")),
                              expression(bold("10")),
                              expression(bold(paste("10"^"2"))),
                              expression(bold(paste("10"^"3"))),
                              expression(bold(paste("10"^"4"))))) +
  scale_x_continuous(breaks=c(-0.2, log10(1),log10(2),log10(3), log10(4),
                              log10(5),log10(7),log10(10),log10(15),
                              log10(20),log10(30)),
                     labels=c(0.6, 1,2,3,4,5,7,10,15,20,30)) + 
  theme(axis.text=element_text(size=15),
        axis.title=element_text(size=15,face="bold"),
        plot.title = element_text(hjust = 0.5),
        text=element_text(family="Helvetica", size=12, face = "bold"),
        panel.background = element_blank(),
        axis.line = element_line(colour = "black")) +
  ggtitle("Kepler data: Mass-Radius Relations") 

pdf.name <- "figure4-KeplerMRrelationWithBootstrap.pdf"
savepdf <- function(pdf.name, Myplot) {
  pdf(file = pdf.name, width = 6, height = 4.5)
  print(Myplot)
  dev.off()
}
savepdf(pdf.name, figure4)


##################### Figure 5 #######################
library(matrixStats)
num.boot <- 100
boot.sd.quantile <- sqrt(rowQuantiles(M.cond.R.var.boot[,1:num.boot], 
                                      probs = c(0.16, 0.5, 0.84)))
sd.df <- data.frame(
  radius = R.points,
  boot.sd.quantile = boot.sd.quantile
)

figure5 <- ggplot()  +
  geom_path(data = sd.df, aes(radius, boot.sd.quantile[,2]),
            color = darkblue, size = 1) +
  geom_ribbon(data = sd.df, 
              aes(x = radius, ymin = boot.sd.quantile[,1], ymax = boot.sd.quantile[,3]),
              fill = darkblue, alpha = 0.3) +
  ylab(expression(atop("Instrinic scatter of M-R relation", 
                       paste("log(Mass (M"["Earth"],"))")))) + 
  # ylab(label = "Intrinsic scatter of M-R relation") +
  xlab(expression(paste("Radius (R"["Earth"],")"))) + 
  coord_cartesian(ylim = c(0.05, 0.35), xlim = c(-0.3, 1.5)) +
  scale_x_continuous(breaks=c(-0.2, log10(1),log10(2),log10(3), log10(4),
                              log10(5),log10(7),log10(10),log10(15),
                              log10(20),log10(30)),
                     labels=c(0.6,1,2,3,4,5,7,10,15,20,30)) + 
  # geom_rect(aes(xmin=-0.07572071, xmax=0.9, ymin=0, ymax=0.6),
  #           colour = "grey10", alpha =  0.1, linetype = 2) +
  geom_vline(xintercept = c(log10(0.86)), linetype = 1, color = "grey50") + 
  geom_vline(xintercept = c(log10(5),  log10(11)), linetype = 2, color = "grey50") + 
  theme(axis.text=element_text(size=15),
        axis.title=element_text(size=15,face="bold"),
        plot.title = element_text(hjust = 0.5),
        text=element_text(family="Helvetica", size=12, face = "bold"),
        panel.background = element_blank(),
        axis.line = element_line(colour = "black"))

pdf.name <- "figure5-Keplersd.pdf"
savepdf(pdf.name, figure5)


##################### Figure 5-a #######################
library(matrixStats)
num.boot <- 100
boot.sd.quantile <- sqrt(rowQuantiles(M.cond.R.var.boot[,1:num.boot], 
                                      probs = c(0.16, 0.5, 0.84)))
sd.trans.df <- data.frame(
  radius = R.points,
  boot.sd.trans.quantile = 10^(boot.sd.quantile)
)

figure5a <- 
  ggplot()  +
  geom_path(data = sd.trans.df, aes(radius, boot.sd.trans.quantile.50.),
            color = darkblue, size = 1) +
  geom_ribbon(data = sd.trans.df, 
              aes(x = radius, 
                  ymin = boot.sd.trans.quantile.16., ymax = boot.sd.trans.quantile.84.),
              fill = darkblue, alpha = 0.3) +
  ylab(expression(paste("Mass (M"["Earth"],")"))) +
  xlab(expression(paste("Radius (R"["Earth"],")"))) + 
  coord_cartesian( xlim = c(-0.3, 1.5)) +
  scale_x_continuous(breaks=c(-0.2, log10(1),log10(2),log10(3), log10(4),
                              log10(5),log10(7),log10(10),log10(15),
                              log10(20),log10(30)),
                     labels=c(0.6,1,2,3,4,5,7,10,15,20,30)) + 
  # geom_rect(aes(xmin=-0.07572071, xmax=0.9, ymin=0, ymax=0.6),
  #           colour = "grey10", alpha =  0.1, linetype = 2) +
  geom_vline(xintercept = c(log10(0.86)), linetype = 1, color = "grey50") + 
  geom_vline(xintercept = c(log10(5),  log10(11)), linetype = 2, color = "grey50") + 
  theme(axis.text=element_text(size=15),
        axis.title=element_text(size=15,face="bold"),
        plot.title = element_text(hjust = 0.5),
        text=element_text(family="Helvetica", size=12, face = "bold"),
        panel.background = element_blank(),
        axis.line = element_line(colour = "black"))

pdf.name <- "figure5a-Keplersd-transformed.pdf"
savepdf(pdf.name, figure5a)


##################### Figure 6 #######################
# plot densities
weights <- weights.boot
deg.choose <- 55

cond.density <- function(y, y.max, y.min, x, x.max, x.min, deg, w.hat) {
  
  # evaluate beta density for y
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
  density.pdf <- sum(w.hat * kronecker(density.indv.pdf, y.beta.indv))
  density <- density.pdf/denominator
  
  return(density)
}

# using the weights to calculate density function at R = log10(1.5)
R_1 <- log10(1)
R_3 <- log10(3)
R_5 <- log10(5)
R_10 <- log10(10)
R_15 <- log10(15)
M.seq <- seq(Mass.min, Mass.max, length.out = 100)
density_R_1 <- density_R_3 <- density_R_5 <-
  density_R_10 <- density_R_15 <- matrix(NA, 100, num.boot)
for (i.boot in 1:num.boot) {
  density_R_1[, i.boot] <- 
    sapply(M.seq, FUN = cond.density, 
           y = R_1, y.max = Radius.max, y.min = Radius.min, 
           x.max = Mass.max, x.min = Mass.min, deg = deg.choose,
           w.hat = weights[, i.boot])
  density_R_3[, i.boot] <- 
    sapply(M.seq, FUN = cond.density, 
           y = R_3, y.max = Radius.max, y.min = Radius.min, 
           x.max = Mass.max, x.min = Mass.min, deg = deg.choose,
           w.hat = weights[, i.boot])
  density_R_5[, i.boot] <- 
    sapply(M.seq, FUN = cond.density, 
           y = R_5, y.max = Radius.max, y.min = Radius.min, 
           x.max = Mass.max, x.min = Mass.min, deg = deg.choose,
           w.hat = weights[, i.boot])
  density_R_10[, i.boot] <- 
    sapply(M.seq, FUN = cond.density, 
           y = R_10, y.max = Radius.max, y.min = Radius.min, 
           x.max = Mass.max, x.min = Mass.min, deg = deg.choose,
           w.hat = weights[, i.boot])
  density_R_15[, i.boot] <- 
    sapply(M.seq, FUN = cond.density, 
           y = R_15, y.max = Radius.max, y.min = Radius.min, 
           x.max = Mass.max, x.min = Mass.min, deg = deg.choose,
           w.hat = weights[, i.boot])
}
density_R_1.quantile <- rowQuantiles(density_R_1, probs = c(0.16, 0.5, 0.84))
density_R_3.quantile <- rowQuantiles(density_R_3, probs = c(0.16, 0.5, 0.84))
density_R_5.quantile <- rowQuantiles(density_R_5, probs = c(0.16, 0.5, 0.84))
density_R_10.quantile <- rowQuantiles(density_R_10, probs = c(0.16, 0.5, 0.84))
density_R_15.quantile <- rowQuantiles(density_R_15, probs = c(0.16, 0.5, 0.84))

R_1.df <- data.frame(
  Mass_1.seq <- M.seq[M.seq<2], median_1 <- density_R_1.quantile[M.seq<2,2],
  lower_1 <- density_R_1.quantile[M.seq<2,1], upper_1 <- density_R_1.quantile[M.seq<2,3]
)

R_3.df <- data.frame(
  Mass_3.seq <- M.seq[M.seq<2 & M.seq > -0.5], 
  median_3 <- density_R_3.quantile[M.seq<2 & M.seq > -0.5,2],
  lower_3 <- density_R_3.quantile[M.seq<2 & M.seq > -0.5,1], 
  upper_3 <- density_R_3.quantile[M.seq<2 & M.seq > -0.5,3]
)

R_5.df <- data.frame(
  Mass_5.seq <- M.seq[M.seq>-0.3 & M.seq < 2.8], 
  median_5 <- density_R_5.quantile[M.seq>-0.3 & M.seq < 2.8,2],
  lower_5 <- density_R_5.quantile[M.seq>-0.3 & M.seq < 2.8,1], 
  upper_5 <- density_R_5.quantile[M.seq>-0.3 & M.seq < 2.8,3]
)

R_10.df <- data.frame(
  Mass_10.seq <- M.seq[M.seq>-0.2], median_10 <- density_R_10.quantile[M.seq>-.2,2],
  lower_10 <- density_R_10.quantile[M.seq>-0.2,1], upper_10 <- density_R_10.quantile[M.seq>-0.2,3]
)

R_15.df <- data.frame(
  Mass_15.seq <- M.seq[M.seq>1], median_15 <- density_R_15.quantile[M.seq>1,2],
  lower_15 <- density_R_15.quantile[M.seq>1,1], upper_15 <- density_R_15.quantile[M.seq>1,3]
)
######## Plot ##############
black <- "#1D1D24"
blue <- "#36527A"
green <- "#416339"
orange <- "#BE4B3E"
red <- "#C145FF"
yellow <- "#FFE5AB"

figure6 <-
  ggplot() + 
  geom_path(data = R_1.df, aes(Mass_1.seq, median_1),
            color = black, size = 0.5) + 
  geom_ribbon(data = R_1.df, 
              aes(x = Mass_1.seq, ymin = lower_1, ymax = upper_1),
              fill = black, alpha = 0.3) +
  geom_path(data = R_3.df, aes(Mass_3.seq, median_3),
            color = blue, size = 0.5) + 
  geom_ribbon(data = R_3.df, 
              aes(x = Mass_3.seq, ymin = lower_3, ymax = upper_3),
              fill = blue, alpha = 0.3) +
  geom_path(data = R_5.df, aes(Mass_5.seq, median_5),
            color = green, size = 0.5) + 
  geom_ribbon(data = R_5.df, 
              aes(x = Mass_5.seq, ymin = lower_5, ymax = upper_5),
              fill = green, alpha = 0.3) +
  geom_path(data = R_10.df, aes(Mass_10.seq, median_10),
            color = orange, size = 0.5) + 
  geom_ribbon(data = R_10.df, 
              aes(x = Mass_10.seq, ymin = lower_10, ymax = upper_10),
              fill = orange, alpha = 0.3) +
  geom_path(data = R_15.df, aes(Mass_15.seq, median_15),
            color = red, size = 0.5) + 
  geom_ribbon(data = R_15.df, 
              aes(x = Mass_15.seq, ymin = lower_15, ymax = upper_15),
              fill = red, alpha = 0.3) +
  coord_cartesian(xlim = c(-1, 4), ylim = c(0, 1.3)) +
  scale_x_continuous(breaks=c(-1, 0, 1, 2, 3, 4),
                     labels=c(expression(bold("0.1")),
                              expression(bold("1")),
                              expression(bold("10")),
                              expression(bold(paste("10"^"2"))),
                              expression(bold(paste("10"^"3"))),
                              expression(bold(paste("10"^"4"))))) +
  theme(axis.text=element_text(size=15),
        axis.title=element_text(size=15,face="bold"),
        plot.title = element_text(hjust = 0.5),
        text=element_text(family="Helvetica", size=12, face = "bold"),
        panel.background = element_rect(fill='white', colour='black', size = 1,
                                        linetype = 1),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        # panel.border = element_rect(linetype = 1, fill = NA) +
        # axis.line = element_line(colour = "black", linetype = 1),
        axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank()) +
    #ylab(label = "Conditional densities \n for masses given radii") +
    # ylab(label = " ") + 
    xlab(expression(paste("Mass (M"["Earth"],")"))) + 
    annotate("text", x=-0.1, y=1.2, size = 4, label= "Radius == 1", 
             parse=TRUE, col = black) +
    annotate("text", x=1, y=1.3, size = 4, label= "Radius == 3", 
             parse=TRUE, col = blue) +
    annotate("text", x=1.5, y=1.05, size = 4, label = "Radius == 5", 
             parse=TRUE, col = green) +
    annotate("text", x=1.8, y=0.9, size = 4, label= "Radius == 10", 
             parse=TRUE, col = orange) +
    annotate("text", x=2.6, y=1.2, size = 4, label= "Radius == 15", 
             parse=TRUE, col = red)

pdf.name <- "figure6-conditionaldensities.pdf"
savepdf(pdf.name, figure6)

##################### Figure 7 #######################
# plot R|M
#R.cond.M.mean <- do.call("cbind", R.cond.M.boot)[, c(1,seq(2,num.boot*2,by = 2))]
#R.cond.M.var <- do.call("cbind", R.cond.M.boot.var)[, c(1,seq(2,num.boot*2,by = 2))]
# orgnize dataset
RgM.nonparam.pred.df <- data.frame(
  mass = M.points,
  mean = R.cond.M, 
  upper = R.cond.M.upper,
  lower = R.cond.M.lower
)

RgM.nonparam.boot.df <- data.frame(
  mass = M.points,
  upper.boot = rowQuantiles(R.cond.M.boot, probs = 0.84),
  lower.boot = rowQuantiles(R.cond.M.boot, probs = 0.16)
)

# Making plot
figure7 <-  ggplot() +
  geom_point(data = data.points, aes(x = lnM.obs, y = lnR.obs), 
             col = "grey10", alpha = 0.6, size = 0.5) +
  geom_errorbar(data = data.points, 
                aes(x = lnM.obs, ymin = lnR.min, ymax = lnR.max),
                col = "grey20", alpha = 0.6, size = 0.15) +
  geom_errorbarh(data = data.points, 
                 aes(x = lnM.obs, y = lnR.obs, xmin = lnM.min, xmax = lnM.max),
                 col = "grey20", alpha = 0.6, size = 0.15) +
  geom_path(data = RgM.nonparam.pred.df, aes(mass, mean),
            color = darkblue, size = 1) +
  geom_ribbon(data = RgM.nonparam.pred.df, 
              aes(x = mass, ymin = lower, ymax = upper),
              fill = darkblue, alpha = 0.3) +
  geom_ribbon(data = RgM.nonparam.boot.df, 
              aes(x = mass, ymin = lower.boot, ymax = upper.boot),
              fill = lightblue, alpha = 0.3) +
  xlab(expression(paste("Mass (M"["Earth"],")"))) +
  ylab(expression(paste("Radius (R"["Earth"],")"))) + 
  coord_cartesian(xlim = c(-1.5,4), ylim = c(-0.3, 1.5)) +
  scale_x_continuous(breaks=c(-1, 0, 1, 2, 3, 4),
                     labels=c(expression(bold("0.1")),
                              expression(bold("1")),
                              expression(bold("10")),
                              expression(bold(paste("10"^"2"))),
                              expression(bold(paste("10"^"3"))),
                              expression(bold(paste("10"^"4"))))) +
  scale_y_continuous(breaks=c(-0.2, log10(1),log10(2),log10(3), 
                              log10(5),log10(10),
                              log10(20),log10(30)),
                     labels=c(0.6, 1,2,3,5,10,20,30)) + 
  theme(axis.text=element_text(size=15),
        axis.title=element_text(size=15,face="bold"),
        plot.title = element_text(hjust = 0.5),
        text=element_text(family="Helvetica", size=12, face = "bold"),
        panel.background = element_blank(),
        axis.line = element_line(colour = "black"))
  #  ggtitle("Kepler data: Radius-Mass Relations") 

pdf.name <- "figure7-KeplerRadiusgivenMasses.pdf"
savepdf(pdf.name, figure7)

########### Figure 8 ##################
# using the weights to calculate density function at R = log10(1.5)
M_1 <- log10(1)
M_10 <- log10(10)
M_50 <- log10(50)
M_100 <- log10(100)
M_500 <- log10(500)
R.seq <- seq(Radius.min, Radius.max, length.out = 100)
density_M_1 <- density_M_10 <- density_M_50 <-
  density_M_100 <- density_M_500 <- matrix(NA, 100, num.boot)
for (i.boot in 1:num.boot) {
  weights.t <- as.vector(t(matrix(weights[, i.boot], deg.choose, deg.choose)))
  
  density_M_1[, i.boot] <- 
    sapply(R.seq, FUN = cond.density, 
           y = M_1, y.max = Mass.max, y.min = Mass.min, 
           x.max = Radius.max, x.min = Radius.min, deg = deg.choose,
           w.hat = weights.t)
  density_M_10[, i.boot] <- 
    sapply(R.seq, FUN = cond.density, 
           y = M_10, y.max = Mass.max, y.min = Mass.min, 
           x.max = Radius.max, x.min = Radius.min, deg = deg.choose,
           w.hat = weights.t)
  density_M_50[, i.boot] <- 
    sapply(R.seq, FUN = cond.density, 
           y = M_50, y.max = Mass.max, y.min = Mass.min, 
           x.max = Radius.max, x.min = Radius.min, deg = deg.choose,
           w.hat = weights.t)
  density_M_100[, i.boot] <- 
    sapply(R.seq, FUN = cond.density, 
           y = M_100, y.max = Mass.max, y.min = Mass.min, 
           x.max = Radius.max, x.min = Radius.min, deg = deg.choose,
           w.hat = weights.t)
  density_M_500[, i.boot] <- 
    sapply(R.seq, FUN = cond.density, 
           y = M_500, y.max = Mass.max, y.min = Mass.min, 
           x.max = Radius.max, x.min = Radius.min, deg = deg.choose,
           w.hat = weights.t)
}
density_M_1.quantile <- rowQuantiles(density_M_1, probs = c(0.16, 0.5, 0.84))
density_M_10.quantile <- rowQuantiles(density_M_10, probs = c(0.16, 0.5, 0.84))
density_M_50.quantile <- rowQuantiles(density_M_50, probs = c(0.16, 0.5, 0.84))
density_M_100.quantile <- rowQuantiles(density_M_100, probs = c(0.16, 0.5, 0.84))
density_M_500.quantile <- rowQuantiles(density_M_500, probs = c(0.16, 0.5, 0.84))

M_1.df <- data.frame(
  Radius_1.seq <- R.seq[R.seq>-0.4 & R.seq < 1.2], 
  median_1 <- density_M_1.quantile[R.seq>-0.4 & R.seq < 1.2,2],
  lower_1 <- density_M_1.quantile[R.seq>-0.4 & R.seq < 1.2,1], 
  upper_1 <- density_M_1.quantile[R.seq>-0.4 & R.seq < 1.2,3]
)

M_10.df <- data.frame(
  Radius_10.seq <- R.seq[R.seq>-0.3&R.seq<1.2], 
  median_10 <- density_M_10.quantile[R.seq>-0.3&R.seq<1.2,2],
  lower_10 <- density_M_10.quantile[R.seq>-0.3&R.seq<1.2,1], 
  upper_10 <- density_M_10.quantile[R.seq>-0.3&R.seq<1.2,3]
)

M_50.df <- data.frame(
  Radius_50.seq <- R.seq[R.seq>0.1], 
  median_50 <- density_M_50.quantile[R.seq>0.1,2],
  lower_50 <- density_M_50.quantile[R.seq>0.1,1], 
  upper_50 <- density_M_50.quantile[R.seq>0.1,3]
)

M_100.df <- data.frame(
  Radius_100.seq <- R.seq[R.seq>0.3], 
  median_100 <- density_M_100.quantile[R.seq>0.3,2],
  lower_100 <- density_M_100.quantile[R.seq>0.3,1], 
  upper_100 <- density_M_100.quantile[R.seq>0.3,3]
)

M_500.df <- data.frame(
  Radius_500.seq <- R.seq[R.seq>0.5], 
  median_500 <- density_M_500.quantile[R.seq>0.5,2],
  lower_500 <- density_M_500.quantile[R.seq>0.5,1], 
  upper_500 <- density_M_500.quantile[R.seq>0.5,3]
)

black <- "#1D1D24"
blue <- "#36527A"
green <- "#416339"
orange <- "#BE4B3E"
red <- "#C145FF"
yellow <- "#FFE5AB"

figure8 <- 
  ggplot() + 
  geom_path(data = M_1.df, aes(Radius_1.seq, median_1),
            color = black, size = 0.5) + 
  geom_ribbon(data = M_1.df, 
              aes(x = Radius_1.seq, ymin = lower_1, ymax = upper_1),
              fill = black, alpha = 0.3) +
  geom_path(data = M_10.df, aes(Radius_10.seq, median_10),
            color = blue, size = 0.5) + 
  geom_ribbon(data = M_10.df, 
              aes(x = Radius_10.seq, ymin = lower_10, ymax = upper_10),
              fill = blue, alpha = 0.3) +
  geom_path(data = M_50.df, aes(Radius_50.seq, median_50),
            color = green, size = 0.5) + 
  geom_ribbon(data = M_50.df, 
              aes(x = Radius_50.seq, ymin = lower_50, ymax = upper_50),
              fill = green, alpha = 0.3) +
  geom_path(data = M_100.df, aes(Radius_100.seq, median_100),
            color = orange, size = 0.5) + 
  geom_ribbon(data = M_100.df, 
              aes(x = Radius_100.seq, ymin = lower_100, ymax = upper_100), 
              fill = orange, alpha = 0.3) +
  geom_path(data = M_500.df, aes(Radius_500.seq, median_500),
            color = red, size = 0.5) + 
  geom_ribbon(data = M_500.df, 
              aes(x = Radius_500.seq, ymin = lower_500, ymax = upper_500),
              fill = red, alpha = 0.3) +
  coord_cartesian(ylim = c(0,5), xlim = c(-0.3, 1.3)) +
  scale_x_continuous(breaks=c(-0.2, log10(1),log10(2),log10(3), log10(4),
                              log10(5),log10(7),log10(10),log10(15),
                              log10(20),log10(30)),
                     labels=c(0.6,1,2,3,4,5,7,10,15,20,30)) + 
  theme(axis.text=element_text(size=15),
        axis.title=element_text(size=15,face="bold"),
        plot.title = element_text(hjust = 0.5),
        text=element_text(family="Helvetica", size=12, face = "bold"),
        panel.background = element_rect(fill='white', colour='black', size = 1,
                                        linetype = 1),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        # panel.border = element_rect(linetype = 1, fill = NA) +
        # axis.line = element_line(colour = "black", linetype = 1),
        axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank()) +
  # ylab(label = "Conditional densities \n for radii given masses") +
  ylab(label = " ") + 
  xlab(expression(paste("Radius (R"["Earth"],")"))) + 
  annotate("text", x=0.05, y=3.1, size = 4, label= "Mass == 1",
           parse=TRUE, col = black) +
  annotate("text", x=0.4, y=2.6, size = 4, label= "Mass == 10",
           parse=TRUE, col = blue) +
  annotate("text", x=0.7, y= 2.0, size = 4, label = "Mass == 50",
           parse=TRUE, col = green) +
  annotate("text", x=0.9, y=3.5, size = 4, label= "Mass == 100",
           parse=TRUE, col = orange) +
  annotate("text", x=1.1, y=5, size = 4, label= "Mass == 500",
           parse=TRUE, col = red)

pdf.name <- "figure8-conditionaldensitiesRM.pdf"
savepdf(pdf.name, figure8)

##################### Figure 9 #######################
k2.raw.data <- read.csv(file = "k2-planets.csv", skip = 31)
k2.data <- subset(k2.raw.data, select = c("rowid", "pl_hostname", "pl_masse",
                                          "pl_masseerr1", "pl_masseerr2",
                                          "pl_rade", "pl_radeerr1", "pl_radeerr2"))
k2.Mass.sigma <- (k2.data$pl_masseerr1 + abs(k2.data$pl_masseerr2))/2
k2.Radius.sigma <- (k2.data$pl_radeerr1 + abs(k2.data$pl_radeerr2))/2
k2.Mass.obs <- k2.data$pl_masse
k2.Radius.obs <- k2.data$pl_rade

k2.lM.obs <- log10(k2.Mass.obs)
k2.lR.obs <- log10(k2.Radius.obs)
k2.lM.sd.obs <- 0.434*k2.Mass.sigma/k2.Mass.obs
k2.lR.sd.obs <- 0.434*k2.Radius.sigma/k2.Radius.obs
k2.data.points <- data.frame(k2.lnM.obs = k2.lM.obs, k2.lnR.obs = k2.lR.obs,
                             k2.lnM.max = k2.lM.obs+k2.lM.sd.obs,
                             k2.lnM.min = k2.lM.obs-k2.lM.sd.obs,
                             k2.lnR.max = k2.lR.obs+k2.lR.sd.obs,
                             k2.lnR.min = k2.lR.obs-k2.lR.sd.obs)

# using k2 planet radius to predict distribution of mass
source("MRpredict/MLE.R")
k2.pred.McR <- sapply(k2.lR.obs, FUN = cond.density.quantile,
                   y.max = Radius.max, y.min = Radius.min,
                   x.max = Mass.max, x.min = Mass.min,
                   deg = deg.choose, w.hat = weights.mle)
k2.pred.mean <- k2.pred.McR[1, ]
k2.pred.lower <- k2.pred.McR[3, ]
k2.pred.upper <- k2.pred.McR[4, ]

k2.pred.df <- data.frame(
  k2.lnR.obs = k2.lR.obs+0.002,
  k2.pred.mean = k2.pred.mean,
  k2.pred.lower = k2.pred.lower,
  k2.pred.upper = k2.pred.upper
)

library(ggplot2)
library(extrafont)
figure9 <- ggplot() +
  geom_point(data = k2.data.points, aes(x = k2.lnR.obs, y = k2.lnM.obs), 
             col = "grey10", alpha = 1, size = 1.5) +
  geom_errorbar(data = k2.data.points, 
                aes(x = k2.lnR.obs, ymin = k2.lnM.min, ymax = k2.lnM.max),
                col = "grey20", alpha = 1, size = 0.3) +
  geom_errorbarh(data = k2.data.points,
                aes(x = k2.lnR.obs, y = k2.lnM.obs,
                    xmin = k2.lnR.min, xmax = k2.lnR.max),
                col = "grey20", alpha = 1, size = 0.3) +
  # geom_point(data = k2.pred.df, aes(x = k2.lnR.obs, y = k2.pred.mean), 
  #            col = "red", shape = 17, size = 2) +
  # geom_errorbar(data = k2.pred.df, 
  #               aes(x = k2.lnR.obs, ymin = k2.pred.lower, ymax = k2.pred.upper),
  #               col = "red", alpha = 0.8, size = 0.3, width=0.02) +
  geom_ribbon(data = nonparam.pred.df, 
              aes(x = radius, ymin = lower, ymax = upper),
              fill = darkblue, alpha = 0.3) +
  xlab(expression(paste("Radius (R"["Earth"],")"))) +
  ylab(expression(paste("Mass (M"["Earth"],")"))) + 
  coord_cartesian(ylim = c(-1.5,4), xlim = c(-0.3, 1.5)) +
  scale_y_continuous(breaks=c(-1, 0, 1, 2, 3, 4),
                     labels=c(expression(bold("0.1")),
                              expression(bold("1")),
                              expression(bold("10")),
                              expression(bold(paste("10"^"2"))),
                              expression(bold(paste("10"^"3"))),
                              expression(bold(paste("10"^"4"))))) +
  scale_x_continuous(breaks=c(-0.2, log10(1),log10(2),log10(3), log10(4),
                              log10(5),log10(7),log10(10),log10(15),
                              log10(20),log10(30)),
                     labels=c(0.6, 1,2,3,4,5,7,10,15,20,30)) + 
  theme(axis.text=element_text(size=15),
        axis.title=element_text(size=15,face="bold"),
        plot.title = element_text(hjust = 0.5),
        text=element_text(family="Helvetica", size=12, face = "bold"),
        panel.background = element_blank(),
        axis.line = element_line(colour = "black")) +
  ggtitle("Predicting K2 planets") 

pdf.name <- "figure9-predictingK2planets.pdf"
savepdf(pdf.name, figure9)


######################### Plot data points ########################
data.o.points <- data.frame(
  M.obs = Mass.obs, 
  R.obs = Radius.obs,
  M.max = Mass.obs+Mass.sigma,
  M.min = Mass.obs-Mass.sigma,
  R.max = Radius.obs+Radius.sigma,
  R.min = Radius.obs-Radius.sigma
)

figure_d1 <- ggplot() +
  geom_point(data = data.o.points, aes(x = R.obs, y = M.obs), 
             col = "grey10", alpha = 1, size = 1) +
  xlab(expression(paste("Radius (R"["Earth"],")"))) +
  ylab(expression(paste("Mass (M"["Earth"],")"))) 
savepdf("kepler-data.pdf", figure_d1)

figure_d2 <- ggplot() + 
  geom_point(data = data.points, aes(x = lnR.obs, y = lnM.obs), 
             col = "grey10", alpha = 1, size = 1) +
  xlab(expression(paste("log(Radius (R"["Earth"],"))"))) +
  ylab(expression(paste("log(Mass (M"["Earth"],"))")))
savepdf("kepler-logdata.pdf", figure_d2)

plot(Radius.obs, Mass.obs)


######################### Plot weights ########################
row <- col <- rep(NA, deg.choose * deg.choose)
row[1:(55*55)] <- (0:54)/(Radius.max - Radius.min) + Radius.min
col[1:(55*55)] <- (0:54)/(Mass.max - Mass.min) + Mass.min

weight.dg <- data.frame(
  row = row,
  col = col,
  weights = weights.mle
)

library(ggplot2)
ggplot() +
  geom_point(data = data.points, aes(x = lnR.obs, y = lnM.obs), 
             col = "grey10", alpha = 0.6, size = 1) +
  # geom_errorbar(data = data.points, 
  #               aes(x = lnR.obs, ymin = lnM.min, ymax = lnM.max),
  #               col = "grey20", alpha = 0.6, size = 0.15) +
  # geom_errorbarh(data = data.points, 
  #                aes(x = lnR.obs, y = lnM.obs, xmin = lnR.min, xmax = lnR.max),
  #                col = "grey20", alpha = 0.6, size = 0.15) +
  geom_ribbon(data = nonparam.pred.df, 
              aes(x = radius, ymin = lower, ymax = upper),
              fill = darkblue, alpha = 0.3) +
  geom_raster(data = weight.dg,
              aes(x = col, y = row, fill = weights), hjust=0.5, vjust=0.5, interpolate=FALSE) + 
  # geom_point(data = weight.dg,
  #             aes(x=row, y=col), 
   #            alpha=0.7, na.rm = T) +
  xlab(expression(paste("Radius (R"["Earth"],")"))) +
  ylab(expression(paste("Mass (M"["Earth"],")"))) + 
  coord_cartesian(ylim = c(-1.5,4), xlim = c(-0.3, 1.5)) +
  scale_y_continuous(breaks=c(-1, 0, 1, 2, 3, 4),
                     labels=c(expression(bold("0.1")),
                              expression(bold("1")),
                              expression(bold("10")),
                              expression(bold(paste("10"^"2"))),
                              expression(bold(paste("10"^"3"))),
                              expression(bold(paste("10"^"4"))))) +
  scale_x_continuous(breaks=c(-0.2, log10(1),log10(2),log10(3), log10(4),
                              log10(5),log10(7),log10(10),log10(15),
                              log10(20),log10(30)),
                     labels=c(0.6, 1,2,3,4,5,7,10,15,20,30)) + 
  theme(axis.text=element_text(size=15),
        axis.title=element_text(size=15,face="bold"),
        plot.title = element_text(hjust = 0.5),
        text=element_text(family="Helvetica", size=12, face = "bold"),
        panel.background = element_blank(),
        axis.line = element_line(colour = "black")) +
  ggtitle("Kepler data: Mass-Radius Relations") 


image(matrix(weights.mle, 55, 55), col=terrain.colors(100),
      xaxs="i", yaxs="i")
abline(h = seq(-0.03, 1.03, length.out = 55), lty = 1, col = "grey30")
abline(v = seq(-0.03, 1.03, length.out = 55), lty = 1, col = "grey30")
lines(R.points/8, (M.cond.R.mean[1:100]+10)/(85), col = "blue", lty = 2)
