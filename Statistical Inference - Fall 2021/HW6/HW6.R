#####################################Question4#######################################################
library(MASS)
library(ggplot2)
absenteeism <- quine
#a)Converting the Eth, Sex, and Lrn variables to binary variables
absenteeism$Eth <- ifelse(absenteeism$Eth == "N",1,0)
absenteeism$Sex <- ifelse(absenteeism$Sex == "M",1,0)
absenteeism$Lrn <- ifelse(absenteeism$Lrn == "SL",1,0)

#b)fitting the model
## Explanatory variables: Eth,Sex,Lrn
##Response variable: Days
absenteeism_mlr <- lm(Days ~ Eth + Sex + Lrn,data = absenteeism)
summary(absenteeism_mlr)

#outputting the adjusted_R_Square
sprintf("Adjusted_R_Squared: %s",summary(absenteeism_mlr)$adj.r.squared)

#Residual Plot
plot(absenteeism_mlr$fitted.values, rstudent(absenteeism_mlr) ,main="residual plot",
     xlab="Predictions(# of absent Days)",ylab="Normalized Residuals")
abline(h=0, lty="dashed")

#checking the Normality of Residuals
qqnorm(rstudent(absenteeism_mlr))
points(qqline(rstudent(absenteeism_mlr)))

#####################################Question8#######################################################
## Response variable:Life.Exp
##Explanatory variables: Population,Income,Illiteracy,Murder,HS.Grad,Frost,Area
df_states <- data.frame(state.x77)
#a)
model1 <- lm(Life.Exp ~ Population + Income + Illiteracy + Murder + HS.Grad + Frost + Area,data = df_states)
summary(model1)

model2 <- lm(Life.Exp ~ Population + Income + Illiteracy + Murder + HS.Grad + Frost ,data = df_states)
summary(model2)

model3 <- lm(Life.Exp ~ Population + Income + Murder + HS.Grad + Frost ,data = df_states)
summary(model3)

model4 <- lm(Life.Exp ~ Population + Murder + HS.Grad + Frost ,data = df_states)
summary(model4)

model5 <- lm(Life.Exp ~ Murder + HS.Grad + Frost ,data = df_states)
summary(model5)

#b)
## Response variable: Life.Exp
##Explanatory variable: Murder
model_b <- lm(Life.Exp ~ Murder ,data = df_states)
summary(model_b)

#c)histogram of residuals
hist(model_b$residuals, xlab = "Residuals", main = "Histogram of Residuals")

##Calculating mean and SD of Residuals
sprintf("Mean of Residuals: %s",mean(model_b$residuals))
sprintf("Standard deviation of Residuals: %s",sd(model_b$residuals))

#d)QQ-plot of residuals vs. Normal distribution with mean=0 and sd = model sd
qqplot(qnorm(ppoints(100),sd = sd(model_b$residuals)), model_b$residuals,
       main = "Residual vs. Normal QQ-plot",xlab = "Normal distribution quantiles",
       ylab = "Residual quantiles")
qqline(model_b$residuals)




















