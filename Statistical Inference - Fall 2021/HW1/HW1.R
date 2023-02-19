
#Question 8 **************************************

#a) creating a vector
nums <- c(57,66,72,78,79,79,81,81,82,83,84,87,88,88,89,90,91,92,94,95)

#b) Calculating Median,Variance, Sd and Mode(s)
#Calculating the Median
med <- median(nums)

#Calculating the Variance
vari <- var(nums)

#Calculating the Standard Deviation
sdev <- sd(nums)

#printing median, variance and sd
sprintf("median: %s", med)
sprintf("variance: %s", vari)
sprintf("standard deviation: %s", sdev)

#Calculating the Mode(s)
y <- table(nums)
df <- data.frame(y)
modes <- df$values[df$Freq == max(df$Freq)]

print(modes)


#c , d) detecting outliers and plotting the box plot

#plotting the box plot of nums
boxnums = boxplot(nums)

#calculating the outliers
#getting the values of upper whisker and lower whisker in the boxnums boxplot
lower_whisker = boxnums$stats[1]
upper_whisker = boxnums$stats[5]

upper_outliers <- nums[nums < lower_whisker]
lower_outliers <- nums[nums > upper_whisker]
 
print(upper_outliers)
print(lower_outliers)

#e) plotting the histogram
hist(nums, freq = FALSE)

#____________________________________________________________________________

#Question 9 **************************************

#a) Identifying the variables and their types
sapply(imdb, typeof)

#b) plotting Bar plot for the number of movies produced each year
barplot(table(imdb$year), main = "Number of movies produced each year", 
        xlab = "year", ylab = "# of movies", ylim = c(0,400), 
        col = colorRampPalette(colors = c("lightblue", "blue"))(21))

#c) plotting the Histogram of USA_gross_income
hist(imdb$USA_gross_income, col = "pink")

#d) side-by-side boxplots (distribution of movie durations | tomatometer_status)
side_box = boxplot(imdb$duration ~ imdb$tomatometer_status, xlab = "tomatometer_status", 
        ylab = "movie duration(minutes)", col=rainbow(3))

#finding the outliers for each tomatometer status

#finding the upper and lower whiskers for each of the boxes
dur_tomat_df <- data.frame(tomatometer_status = c("Certified-Fresh","Fresh","Rotten"),
                           lower_whisker = c(side_box$stats[1,1:3]),
                           upper_whisker = c(side_box$stats[5,1:3]))

##finding the outliers for "Certified_Fresh"
outliers_Certified_Fresh <- imdb$duration[imdb$tomatometer_status == "Certified-Fresh"]
upper_outliers_Certified_Fresh <- 
  outliers_Certified_Fresh[outliers_Certified_Fresh > dur_tomat_df$upper_whisker[dur_tomat_df$tomatometer_status == "Certified-Fresh"]]
lower_outliers_Certified_Fresh <- 
  outliers_Certified_Fresh[outliers_Certified_Fresh < dur_tomat_df$lower_whisker[dur_tomat_df$tomatometer_status == "Certified-Fresh"]]
print(lower_outliers_Certified_Fresh)
print(upper_outliers_Certified_Fresh)

##finding the outliers for "Fresh"
outliers_Fresh <- imdb$duration[imdb$tomatometer_status == "Fresh"]
upper_outliers_Fresh <- 
  outliers_Fresh[outliers_Fresh > dur_tomat_df$upper_whisker[dur_tomat_df$tomatometer_status == "Fresh"]]
lower_outliers_Fresh <- 
  outliers_Fresh[outliers_Fresh < dur_tomat_df$lower_whisker[dur_tomat_df$tomatometer_status == "Fresh"]]
print(lower_outliers_Fresh)
print(upper_outliers_Fresh)

##finding the outliers for "Rotten"
outliers_Rotten <- imdb$duration[imdb$tomatometer_status == "Rotten"]
print(outliers_Rotten)
upper_outliers_Rotten <- 
  outliers_Rotten[outliers_Rotten > dur_tomat_df$upper_whisker[dur_tomat_df$tomatometer_status == "Rotten"]]
lower_outliers_Rotten <- 
  outliers_Rotten[outliers_Rotten < dur_tomat_df$lower_whisker[dur_tomat_df$tomatometer_status == "Rotten"]]
print(lower_outliers_Rotten)
print(upper_outliers_Rotten)


#e) Categorizing the movies based on their durations into 4 groups and plotting pie chart
imdb$duration_category <-
    ifelse(imdb$duration > 200, "very long",
           ifelse(imdb$duration > 150, "long",
                  ifelse(imdb$duration > 80, "standard","short")))
#getting the frequency of movies in each category
very_long_freq <- length(imdb$imdb_ID[imdb$duration_category == "very long"])
long_freq <- length(imdb$imdb_ID[imdb$duration_category == "long"])
standard_freq <- length(imdb$imdb_ID[imdb$duration_category == "standard"])
short_freq <- length(imdb$imdb_ID[imdb$duration_category == "short"])

#setting the arguments of the chart
x <- c(very_long_freq,long_freq,standard_freq,short_freq)
labels <- c("very long","long","standard","short")

piepercent<- round(100*x/sum(x), 1)
piepercent <- as.character(piepercent)
piepercent <- paste(piepercent, "%", sep="")

# Plotting the pie chart
pie(x, labels = piepercent, main = "Movie duration pie chart",col = rainbow(length(x)))
legend("topright", c("very long","long","standard","short"), cex = 0.8,
       fill = rainbow(length(x)))

#f) scatter plot for determining the relationship "USA_gross_income" and "worldwide_gross_income"
plot(imdb$USA_gross_income,imdb$worldwide_gross_income, xlab = "USA_gross_income",
     ylab = "worldwide_gross_income")


