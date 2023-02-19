#####################################Question6#######################################################
#a)H0: p=1/26 , HA: p > 1/26
#b)
#calculating the expected values
correct_guess_observed <- 110
wrong_guess_observed <- 210
correct_guess_expected <- 110*(1/26)
wrong_guess_expected <- 210*(1/26)
#calculating the chi-statistic
chi_square_statistic <- (correct_guess_observed - correct_guess_expected)^2/correct_guess_expected +
  (wrong_guess_observed - wrong_guess_expected)^2/wrong_guess_expected
#calculating degree of freedom
df <- 2-1
##calculating the p-value
p_value <- 0
p_value <- pchisq(chi_square_statistic,df,lower.tail = FALSE)

#outputting the result of the chi_square test
if(p_value < 0.05){
  sprintf("P_value:%s < 0.05  -> mothers don't recognize their child's smell",p_value)
}else{
  sprintf("P_value:%s > 0.05  -> mothers recognize their child's smell",p_value)
}

#####################################Question7#######################################################
##first approach
#loading the data set
library(dplyr, warn.conflicts = FALSE)
library(MASS)
data(caith)
#adding  a "total" column and row to caith
df_caith <- caith
df_caith <- df_caith %>% mutate(total = rowSums(across(where(is.numeric))))
df_caith["total" ,] <- colSums(df_caith[1:4,])

#calculating the Expected Counts table as a dataframe
df_caith_expected <-caith
for(i in 1:(length(df_caith$fair)-1)){
  for (j in 1:(length(df_caith)-1)) {
    df_caith_expected[i,j]<- df_caith[i,"total"]*df_caith["total",j]/df_caith["total","total"]
  }
}
print(df_caith_expected)
#calculating the chi-square statistic
chi_square <- 0
for(i in 1:(length(df_caith$fair)-1)){
  for (j in 1:(length(df_caith)-1)) {
    chi_square <- chi_square + 
      (df_caith[i,j] - df_caith_expected[i,j])^2/df_caith_expected[i,j]
  }
}
print(chi_square)

#calculating the degree of freedom
df <- (length(df_caith_expected$fair)-1)*((length(df_caith_expected)-1))

#calculating the p-value
p_value <- 0
p_value <- pchisq(chi_square,df,lower.tail = FALSE)

#outputting the result of the chi_square independence test
if(p_value < 0.05){
  sprintf("P_value:%s < 0.05  ->  Reject H0: eye color and hair color aren't independent",p_value)
}else{
  sprintf("P_value:%s > 0.05  ->  Fail to Reject H0: eye color is independent of hair color",p_value)
}



##second approach
chisq_test <- chisq.test(caith)
chisq_test
chisq_test$expected


#####################################Question8#######################################################
#H0: p=0.5 , HA: p > 0.5 , p_observed = 1

#simulating 100 times(each simulation 20 coin tosses) and calculating p_value
p_value <- 0
sum_p_value <- 0
set.seed(194830)
for (i in 1:100) {
  p_sim <- sample(c(0,1), replace=TRUE, size=20)
  proportion_sim <- sum(p_sim[TRUE])/20
  if(proportion_sim == 1){
    sum_p_value <- sum_p_value + 1
  }
}
p_value <- sum_p_value/100

#outputting the result of the simulation
if(p_value < 0.05){
  sprintf("P_value:%s < 0.05  ->  Reject H0: computer systems get damaged in more than 110 degrees",p_value)
}else{
  sprintf("P_value:%s > 0.05  ->  Fail to Reject H0: computer systems don't get damaged in more than 110 degrees",p_value)
}


