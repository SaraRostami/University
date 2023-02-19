library(palmerpenguins)
library(tidyverse)
library(ggplot2)
library(car)
library("lattice")
library("report")
#####################################Question3#######################################################

#a)construscting a dataframe with only "flipper_length_mm" & "species" columns of the penguin dataset
penguins_2col <- data.frame(flipper_length=penguins$flipper_length_mm,species=penguins$species)

#b)plotting flipper length of each species
ggplot(penguins_2col) +
  aes(x = species, y = flipper_length, color = species) +
  geom_jitter() +
  theme(legend.position = "none")

#c)
res_aov <- aov(flipper_length ~ species,data = penguins_2col)
##checking the "Normality" assumption
par(mfrow = c(1, 2))
# histogram
hist(res_aov$residuals)
# QQ-plot
qqPlot(res_aov$residuals,id = FALSE)

#d)checking the "equality of variances" assumption
# Dot plot
dotplot(flipper_length ~ species,data = penguins_2col)

#e)mean and Sd of each group
aggregate(flipper_length ~ species,
          data = penguins_2col,
          function(x) round(c(mean = mean(x), sd = sd(x)), 2))

#f)Doing ANOVA test and showing the results
res_aov <- aov(flipper_length ~ species,data = penguins_2col)
report(res_aov)

#g)


#####################################Question10#######################################################
#a)choosing a numerical variable to base the hypothesis on -> car_train$price
##i)
#H0:mu = 22000
#HA:mu != 22000
null_value = 22000
sample_t <- sample(car_train$price,25)
sample_mean_price <- mean(sample_t)
t_24 <- abs((sample_mean_price - null_value)/(sqrt(var(sample_t))/(25^0.5)))
p_value <- 2 * pt(t_24 , df = 24 , lower.tail = FALSE)
if(p_value < 0.05){
  cat("Reject H0 -> there is a statisticaly significant difference between\n 
        the mean value of 22000 and the sample mean value")
}else{
  cat("Fail to Reject H0 -> the difference between the means is simply due 
      to sampling variability.\nwe can't say the sample mean is not equal to 22000")
}

##ii)making the 95% confidence interval (from,to)
t_star <- abs(qt(0.025 , df = 24))
from <- sample_mean_price - abs(t_star) * (sqrt(var(sample_t))/(25^0.5))
to <- sample_mean_price + abs(t_star) * (sqrt(var(sample_t))/(25^0.5))
if(null_value < from || null_value > to){
  print("Reject H0 -> there is a statisticaly significant difference between 
        the actual mean value and the sample mean value")
  sprintf("%s is not in the confidence interval range(%s,%s)",null_value,from,to)
}else{
  print("Fail to Reject H0 -> the difference between the means is simply due to sampling variability")
  sprintf("%s is in the confidence interval range(%s,%s)",null_value,from,to)
}

##iii)calculating the power
actual_mean_price <- mean(car_train$price,na.rm = TRUE)
se <- sqrt(var(sample_t))/(25^0.5)
t_alpha <- abs(qt(0.025 , df = 24))
t_val_min <- null_value - t_alpha*se
t_val_max <- null_value + t_alpha*se
p1 <- pt((t_val_min - actual_mean_price)/se, df=24)
p2 = pt((t_val_max - actual_mean_price)/se, df=24, lower.tail = FALSE)
power <- P0 + p2
print(P0 + p2)