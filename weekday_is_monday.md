ST558\_PROJECT2
================
Qiaozhi Bao
2020/10/6

# Introduction

## Describe the data

The [Online News Popularity data
set](%22https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity%22)
was published two years ago to summarize a heterogeneous set of features
about articles published by Mashable in a period of two years. There are
61 variables in total from the data set above: 58 predictive attributes,
2 non-predictive and 1 goal field.More details and summarization will be
discussed later in this project.

## The purpose of Analysis

The purpose of this analysis is to create two models(ensemble and not
ensemble) to generate the best predict of the response
attribute–shares.Our analysis will help to determine what kind of
content would be most popular.

## Methods

For this project,I first split the data into training set and test
set,then I examine the data with summary statistics and correlation
plots to see the relationships between predictive attributes and the
relationship between predictive attributes and response variables,then
some meaningless variables were moved. I then utilized the caret package
to create two models.Tree-based model chosen using leave one out cross
validation.Boosted tree model chosen using cross-validation.

# Data Study

## Description of the Used Data

``` r
# Load all libraries
library(tidyverse)
library(ggplot2)
library(randomForest)
library(caret)
library(tree)
library(gbm)
library(corrplot)
library(e1071)
set.seed(1)
```

``` r
# Read in data and removing the first two columns as they are not predictive variables.
news_pop <- read_csv('./OnlineNewsPopularity.csv') %>% select(-`url`,-`timedelta`)
```

    ## Parsed with column specification:
    ## cols(
    ##   .default = col_double(),
    ##   url = col_character()
    ## )

    ## See spec(...) for full column specifications.

``` r
# First to see Monday data
Mon_data <- news_pop%>% filter(weekday_is_monday==1)
Mon_data <- Mon_data %>% select(!starts_with('weekday_is'))
# Check if we have missing values, answer is 'No'
sum(is.na(Mon_data))
```

    ## [1] 0

``` r
Mon_data
```

As there is no missing value in our Monday data, we will step to split
data. By using sample(), with 70% of the data goes to the training set
(4,662 observations, Mon\_train) and 30% goes to the test set (1,999
observations, Mon\_test).

``` r
# Split Monday data,70% for training set and 30% for test set
set.seed(1)
train <- sample(1:nrow(Mon_data),size = nrow(Mon_data)*0.7)
test <- dplyr::setdiff(1:nrow(Mon_data),train)
train_data <-Mon_data[train,]
test_data <- Mon_data[test,]
```

# Data Summarizations

## Predictor Variables

I used the summary() function to calculate summary statistics for each
of the quantitative variables in Mon\_data.

``` r
summary(train_data)
```

    ##  n_tokens_title  n_tokens_content n_unique_tokens  n_non_stop_words
    ##  Min.   : 2.00   Min.   :   0.0   Min.   :0.0000   Min.   :0.0000  
    ##  1st Qu.: 9.00   1st Qu.: 248.0   1st Qu.:0.4738   1st Qu.:1.0000  
    ##  Median :10.00   Median : 397.5   Median :0.5427   Median :1.0000  
    ##  Mean   :10.42   Mean   : 538.2   Mean   :0.5308   Mean   :0.9691  
    ##  3rd Qu.:12.00   3rd Qu.: 711.0   3rd Qu.:0.6088   3rd Qu.:1.0000  
    ##  Max.   :18.00   Max.   :7764.0   Max.   :1.0000   Max.   :1.0000  
    ##  n_non_stop_unique_tokens   num_hrefs      num_self_hrefs      num_imgs     
    ##  Min.   :0.0000           Min.   :  0.00   Min.   : 0.000   Min.   : 0.000  
    ##  1st Qu.:0.6287           1st Qu.:  4.00   1st Qu.: 1.000   1st Qu.: 1.000  
    ##  Median :0.6939           Median :  7.00   Median : 3.000   Median : 1.000  
    ##  Mean   :0.6728           Mean   : 10.62   Mean   : 3.367   Mean   : 4.382  
    ##  3rd Qu.:0.7544           3rd Qu.: 13.00   3rd Qu.: 4.000   3rd Qu.: 3.000  
    ##  Max.   :1.0000           Max.   :162.00   Max.   :51.000   Max.   :93.000  
    ##    num_videos     average_token_length  num_keywords    data_channel_is_lifestyle
    ##  Min.   : 0.000   Min.   :0.000        Min.   : 1.000   Min.   :0.00000          
    ##  1st Qu.: 0.000   1st Qu.:4.475        1st Qu.: 6.000   1st Qu.:0.00000          
    ##  Median : 0.000   Median :4.656        Median : 7.000   Median :0.00000          
    ##  Mean   : 1.367   Mean   :4.536        Mean   : 7.153   Mean   :0.04719          
    ##  3rd Qu.: 1.000   3rd Qu.:4.840        3rd Qu.: 9.000   3rd Qu.:0.00000          
    ##  Max.   :74.000   Max.   :6.513        Max.   :10.000   Max.   :1.00000          
    ##  data_channel_is_entertainment data_channel_is_bus data_channel_is_socmed
    ##  Min.   :0.0000                Min.   :0.0000      Min.   :0.00000       
    ##  1st Qu.:0.0000                1st Qu.:0.0000      1st Qu.:0.00000       
    ##  Median :0.0000                Median :0.0000      Median :0.00000       
    ##  Mean   :0.2059                Mean   :0.1695      Mean   :0.05277       
    ##  3rd Qu.:0.0000                3rd Qu.:0.0000      3rd Qu.:0.00000       
    ##  Max.   :1.0000                Max.   :1.0000      Max.   :1.00000       
    ##  data_channel_is_tech data_channel_is_world   kw_min_min       kw_max_min    
    ##  Min.   :0.0000       Min.   :0.0000        Min.   : -1.00   Min.   :     0  
    ##  1st Qu.:0.0000       1st Qu.:0.0000        1st Qu.: -1.00   1st Qu.:   441  
    ##  Median :0.0000       Median :0.0000        Median : -1.00   Median :   651  
    ##  Mean   :0.1836       Mean   :0.2072        Mean   : 26.82   Mean   :  1231  
    ##  3rd Qu.:0.0000       3rd Qu.:0.0000        3rd Qu.:  4.00   3rd Qu.:  1000  
    ##  Max.   :1.0000       Max.   :1.0000        Max.   :318.00   Max.   :298400  
    ##    kw_avg_min        kw_min_max       kw_max_max       kw_avg_max       kw_min_avg    
    ##  Min.   :   -1.0   Min.   :     0   Min.   :     0   Min.   :     0   Min.   :  -1.0  
    ##  1st Qu.:  136.2   1st Qu.:     0   1st Qu.:843300   1st Qu.:173315   1st Qu.:   0.0  
    ##  Median :  230.5   Median :  1400   Median :843300   Median :242336   Median : 994.2  
    ##  Mean   :  317.1   Mean   : 11822   Mean   :748229   Mean   :257156   Mean   :1086.4  
    ##  3rd Qu.:  352.6   3rd Qu.:  7200   3rd Qu.:843300   3rd Qu.:330765   3rd Qu.:1986.1  
    ##  Max.   :29946.9   Max.   :690400   Max.   :843300   Max.   :798220   Max.   :3602.1  
    ##    kw_max_avg       kw_avg_avg    self_reference_min_shares self_reference_max_shares
    ##  Min.   :     0   Min.   :    0   Min.   :     0            Min.   :     0           
    ##  1st Qu.:  3531   1st Qu.: 2355   1st Qu.:   659            1st Qu.:  1100           
    ##  Median :  4255   Median : 2832   Median :  1200            Median :  2800           
    ##  Mean   :  5582   Mean   : 3074   Mean   :  3951            Mean   :  9970           
    ##  3rd Qu.:  5938   3rd Qu.: 3535   3rd Qu.:  2600            3rd Qu.:  7900           
    ##  Max.   :298400   Max.   :33536   Max.   :690400            Max.   :843300           
    ##  self_reference_avg_sharess   is_weekend     LDA_00            LDA_01       
    ##  Min.   :     0             Min.   :0    Min.   :0.01818   Min.   :0.01819  
    ##  1st Qu.:  1000             1st Qu.:0    1st Qu.:0.02517   1st Qu.:0.02504  
    ##  Median :  2168             Median :0    Median :0.03341   Median :0.03337  
    ##  Mean   :  6321             Mean   :0    Mean   :0.18670   Mean   :0.15456  
    ##  3rd Qu.:  5200             3rd Qu.:0    3rd Qu.:0.24603   3rd Qu.:0.17145  
    ##  Max.   :690400             Max.   :0    Max.   :0.91999   Max.   :0.91997  
    ##      LDA_02            LDA_03            LDA_04        global_subjectivity
    ##  Min.   :0.01819   Min.   :0.01819   Min.   :0.01818   Min.   :0.0000     
    ##  1st Qu.:0.02857   1st Qu.:0.02857   1st Qu.:0.02857   1st Qu.:0.3951     
    ##  Median :0.04000   Median :0.04000   Median :0.04001   Median :0.4512     
    ##  Mean   :0.21064   Mean   :0.21781   Mean   :0.23029   Mean   :0.4402     
    ##  3rd Qu.:0.32402   3rd Qu.:0.35340   3rd Qu.:0.39356   3rd Qu.:0.5047     
    ##  Max.   :0.92000   Max.   :0.91998   Max.   :0.92708   Max.   :1.0000     
    ##  global_sentiment_polarity global_rate_positive_words global_rate_negative_words
    ##  Min.   :-0.38021          Min.   :0.00000            Min.   :0.000000          
    ##  1st Qu.: 0.05543          1st Qu.:0.02820            1st Qu.:0.009674          
    ##  Median : 0.11732          Median :0.03817            Median :0.015303          
    ##  Mean   : 0.11631          Mean   :0.03900            Mean   :0.016784          
    ##  3rd Qu.: 0.17457          3rd Qu.:0.04975            3rd Qu.:0.021818          
    ##  Max.   : 0.55455          Max.   :0.12139            Max.   :0.086168          
    ##  rate_positive_words rate_negative_words avg_positive_polarity min_positive_polarity
    ##  Min.   :0.0000      Min.   :0.0000      Min.   :0.0000        Min.   :0.00000      
    ##  1st Qu.:0.6000      1st Qu.:0.1852      1st Qu.:0.3052        1st Qu.:0.05000      
    ##  Median :0.7059      Median :0.2857      Median :0.3586        Median :0.10000      
    ##  Mean   :0.6779      Mean   :0.2910      Mean   :0.3540        Mean   :0.09543      
    ##  3rd Qu.:0.8000      3rd Qu.:0.3871      3rd Qu.:0.4121        3rd Qu.:0.10000      
    ##  Max.   :1.0000      Max.   :1.0000      Max.   :1.0000        Max.   :1.00000      
    ##  max_positive_polarity avg_negative_polarity min_negative_polarity max_negative_polarity
    ##  Min.   :0.000         Min.   :-1.0000       Min.   :-1.0000       Min.   :-1.000       
    ##  1st Qu.:0.600         1st Qu.:-0.3306       1st Qu.:-0.7000       1st Qu.:-0.125       
    ##  Median :0.800         Median :-0.2510       Median :-0.5000       Median :-0.100       
    ##  Mean   :0.757         Mean   :-0.2581       Mean   :-0.5198       Mean   :-0.106       
    ##  3rd Qu.:1.000         3rd Qu.:-0.1833       3rd Qu.:-0.3000       3rd Qu.:-0.050       
    ##  Max.   :1.000         Max.   : 0.0000       Max.   : 0.0000       Max.   : 0.000       
    ##  title_subjectivity title_sentiment_polarity abs_title_subjectivity
    ##  Min.   :0.0000     Min.   :-1.00000         Min.   :0.0000        
    ##  1st Qu.:0.0000     1st Qu.: 0.00000         1st Qu.:0.1500        
    ##  Median :0.1333     Median : 0.00000         Median :0.5000        
    ##  Mean   :0.2771     Mean   : 0.06694         Mean   :0.3391        
    ##  3rd Qu.:0.5000     3rd Qu.: 0.13636         3rd Qu.:0.5000        
    ##  Max.   :1.0000     Max.   : 1.00000         Max.   :0.5000        
    ##  abs_title_sentiment_polarity     shares      
    ##  Min.   :0.000                Min.   :     4  
    ##  1st Qu.:0.000                1st Qu.:   913  
    ##  Median :0.000                Median :  1400  
    ##  Mean   :0.153                Mean   :  3641  
    ##  3rd Qu.:0.250                3rd Qu.:  2700  
    ##  Max.   :1.000                Max.   :652900

``` r
correlation1 <- cor(train_data[,c(1:10,52)])
corrplot(correlation1,type='upper',tl.pos = 'lt')
corrplot(correlation1,type='lower',method = 'number',add = T,diag = F,tl.pos = 'n')
```

![](weekday_is_monday_files/figure-gfm/unnamed-chunk-17-1.png)<!-- -->

``` r
correlation2 <- cor(train_data[,c(11:20,52)])
corrplot(correlation2,type='upper',tl.pos = 'lt')
corrplot(correlation2,type='lower',method = 'number',add = T,diag = F,tl.pos = 'n')
```

![](weekday_is_monday_files/figure-gfm/unnamed-chunk-17-2.png)<!-- -->

``` r
correlation3 <- cor(train_data[,c(21:30,52)])
```

    ## Warning in cor(train_data[, c(21:30, 52)]): the standard deviation is zero

``` r
corrplot(correlation3,type='upper',tl.pos = 'lt')
corrplot(correlation3,type='lower',method = 'number',add = T,diag = F,tl.pos = 'n')
```

![](weekday_is_monday_files/figure-gfm/unnamed-chunk-17-3.png)<!-- -->

``` r
correlation4 <- cor(train_data[,c(31:40,52)])
corrplot(correlation4,type='upper',tl.pos = 'lt')
corrplot(correlation4,type='lower',method = 'number',add = T,diag = F,tl.pos = 'n')
```

![](weekday_is_monday_files/figure-gfm/unnamed-chunk-17-4.png)<!-- -->

``` r
correlation5 <- cor(train_data[,c(41:51,52)])
corrplot(correlation5,type='upper',tl.pos = 'lt')
corrplot(correlation5,type='lower',method = 'number',add = T,diag = F,tl.pos = 'n')
```

![](weekday_is_monday_files/figure-gfm/unnamed-chunk-17-5.png)<!-- -->
From the correlation plot,I decided to remove some meaningless
variables:`kw_min_min`,`kw_avg_min`,`kw_min_avg`,`is_weekend` Also some
highly correlated variables will be removed too,then we will get a new
train set and test set.

``` r
train_data <- train_data %>% select(!starts_with("LDA"),-is_weekend)
test_data <- test_data %>% select(!starts_with("LDA"),-is_weekend)
train_data <- train_data %>% select(!starts_with('kw'))
test_data <- train_data %>% select(!starts_with('kw'))
```

# First Model

## Tree based model chosen using leave one out cross validation

``` r
tree.method <- train(shares ~.,data = train_data,method='rpart',
                       preProcess = c("center","scale"),
                     trControl = trainControl(method ='LOOCV'))
tree.method$results
tree.method$bestTune
```

# Second Model

## Boosted tree model chosen using cross-validation

``` r
# We will fit the model using repeated CV
boosted.method <- train(shares ~.,data = train_data,method = 'gbm',
                      trControl = trainControl(method = 'repeatedcv', number=5,repeats =2),
                      preProcess = c("center","scale"),
                      verbose = FALSE)

boosted.method$results
boosted.method$bestTune
```

``` r
# predict values on test set and compare RMSE for two models
pred.tree <- predict(tree.method,test_data)
pred.boost <- predict(boosted.method,test_data)
compare <- rbind(postResample(pred.tree,test_data$shares),postResample(pred.boost,test_data$shares))
rownames(compare)<-c("Tree method","Boosted method")
compare
```

    ##                    RMSE  Rsquared      MAE
    ## Tree method    13264.25        NA 3685.593
    ## Boosted method 12372.45 0.1711358 3569.803

From the result we can see the boosted method generates smaller RMSE
which is the same as we expected.
