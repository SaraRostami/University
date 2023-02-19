library(tidyverse)
library(ggmosaic)

#####################################Question3#######################################################
#c)calculating the chance of Benedict Selling an even number of pizzas
lim <- qpois(0.9999999999999999,20)
res <- 0
for (i in seq(0,lim, by= 2)) {
  res <- res + dpois(i,20)
}
print(res)
-----------------------------------------------------------------------------------------------------
#####################################Question8#######################################################

#a)Drawing the Histogram of pricePerServing with 30 bins(recommended by R)
ggplot(Foods, aes(x=pricePerServing, y = ..density..)) + 
  geom_histogram(bins= 30, fill= "pink", col="black")+
  geom_density(col ="darkGreen")+ xlim(0,75)+
  ggtitle("Histogram Of pricePerServing")+ theme_minimal()

#b)Drawing the 2D density plot
ggplot(Foods, aes(x = readyInMinutes, y = healthScore, fill = ..level..)) +
  stat_density_2d(geom = "polygon")

#c)Sorting the categories in the dishType and Drawing a horizontal bar plot
sorted <- data.frame(sort(table(Foods$dishType),increasing = TRUE))
colnames(sorted) <- c("DishType","Count")

ggplot(sorted, aes(x= DishType, y=Count , fill=DishType)) + 
  geom_bar(stat = "identity" ) +
  scale_fill_brewer(palette="Dark2")+ coord_flip()+
  theme(legend.position = "none")+
  ggtitle("Frequency of different types of Dishes")

#d)side-by-side boxplots for different dish types
ggplot(Foods, aes(x = dishType, y = healthScore , fill = dishType)) + 
  geom_boxplot()+scale_fill_brewer(palette="Set2")+
  theme(legend.position = "none")+
  ggtitle("HealthScore of different types of Dishes")

#e)Drawing mosaic plots
ggplot(data = Foods) +
  geom_mosaic(aes(x = product(veryHealthy, dairyFree), fill = veryHealthy)) +   
  labs(y="dairyFree", x="veryHealthy", title = "Mosaic Plot") +
  scale_y_continuous(labels=scales::percent)


