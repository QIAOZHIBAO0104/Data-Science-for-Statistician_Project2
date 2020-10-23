ST558\_PROJECT2
================
Qiaozhi Bao
2020/10/6

# Introduction

## The purpose of Analysis

The [Online News Popularity data
set](%22https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity%22)
was published two years ago to summarize a heterogeneous set of features
about articles published by Mashable in a period of two years. There are
61 variables in total from the data set above: 58 predictive attributes,
2 non-predictive and 1 goal field.More details and summarization will be
discussed later in this project.

The purpose of this analysis is to create three models based on tree based method,boosted tree model and linear model to generate the best prediction of the response attribute ``shares``,Our analysis will help to determine what kind of content would be most popular.

# Packages
tidyverse
ggplot2
randomForest
caret
tree
gbm
corrplot
e1071

# Analysis 
[Sunday](https://github.com/QIAOZHIBAO0104/ST558-Project2/blob/main/weekday_is_sunday.md)
[Monday](https://github.com/QIAOZHIBAO0104/ST558-Project2/blob/main/weekday_is_monday.md)
[Tuesday](https://github.com/QIAOZHIBAO0104/ST558-Project2/blob/main/weekday_is_tuesday.md)
[Thrusday](https://github.com/QIAOZHIBAO0104/ST558-Project2/blob/main/weekday_is_thursday.md)
[Friday](https://github.com/QIAOZHIBAO0104/ST558-Project2/blob/main/weekday_is_friday.md)
[Saturday](https://github.com/QIAOZHIBAO0104/ST558-Project2/blob/main/weekday_is_saturday.md)


# Automation Code

news_pop <- read_csv('./OnlineNewsPopularity.csv') %>% select(-`url`,-`timedelta`)
data.frame(output_file ="MondayAnalysis.md",params = list(weekday="weekday_is_Monday"))
weekdays <- c("weekday_is_monday", "weekday_is_tuesday", "weekday_is_wednesday",
              "weekday_is_thursday", "weekday_is_friday", "weekday_is_saturday",
              "weekday_is_sunday")
output_file <- paste0(weekdays, ".md")
params =lapply(weekdays,FUN=function(x)(list(weekday=x)))
reports <- tibble(output_file, params)
reports
library(rmarkdown)
apply(reports, MARGIN = 1,
      FUN = function(x){
        render(input = "README.Rmd", output_file = x[[1]], params = x[[2]])
      })
      
# Conclusion
Linear model is the optimal model as it generates the smallest RMSE on test set.
