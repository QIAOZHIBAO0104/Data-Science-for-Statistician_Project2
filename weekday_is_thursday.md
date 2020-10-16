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

As our study intention is to predict the popularity of an article, so we
choose the shares as the response variable.After plotting the
correlations between variables, we removed some high related predictive
variables. The two models were fitted by remaining variables in the
training set.

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
params$weekday
```

    ## [1] "weekday_is_thursday"

``` r
# First to see Monday data
data <- news_pop%>% select(!starts_with('weekday_is'),params$weekday)
# Check if we have missing values, answer is 'No'
sum(is.na(data))
```

    ## [1] 0

``` r
data <-data %>% filter(data[,53]==1) %>%select(-params$weekday)
```

As there is no missing value in our Monday data, we will step to split
data. By using sample(), with 70% of the data goes to the training set
(4,662 observations, Mon\_train) and 30% goes to the test set (1,999
observations, Mon\_test).

``` r
# Split Monday data,70% for training set and 30% for test set
set.seed(1)
train <- sample(1:nrow(data),size = nrow(data)*0.7)
test <- dplyr::setdiff(1:nrow(data),train)
train_data <-data[train,]
test_data <- data[test,]
```

# Data Summarizations

## Predictor Variables

I used the `summary()` function to calculate summary statistics for each
of the quantitative variables in data.I divided the data into trunks to
make plots easier to compare.

``` r
summary(train_data)
```

    ##  n_tokens_title  n_tokens_content n_unique_tokens  n_non_stop_words n_non_stop_unique_tokens
    ##  Min.   : 3.00   Min.   :   0.0   Min.   :0.0000   Min.   :0.0000   Min.   :0.0000          
    ##  1st Qu.: 9.00   1st Qu.: 245.0   1st Qu.:0.4720   1st Qu.:1.0000   1st Qu.:0.6260          
    ##  Median :10.00   Median : 398.5   Median :0.5408   Median :1.0000   Median :0.6923          
    ##  Mean   :10.33   Mean   : 538.8   Mean   :0.5314   Mean   :0.9699   Mean   :0.6737          
    ##  3rd Qu.:12.00   3rd Qu.: 697.8   3rd Qu.:0.6093   3rd Qu.:1.0000   3rd Qu.:0.7556          
    ##  Max.   :18.00   Max.   :4585.0   Max.   :0.9545   Max.   :1.0000   Max.   :1.0000          
    ##    num_hrefs      num_self_hrefs      num_imgs         num_videos    average_token_length
    ##  Min.   :  0.00   Min.   : 0.000   Min.   :  0.000   Min.   : 0.00   Min.   :0.000       
    ##  1st Qu.:  4.00   1st Qu.: 1.000   1st Qu.:  1.000   1st Qu.: 0.00   1st Qu.:4.483       
    ##  Median :  7.00   Median : 2.000   Median :  1.000   Median : 0.00   Median :4.674       
    ##  Mean   : 10.56   Mean   : 3.131   Mean   :  4.413   Mean   : 1.25   Mean   :4.550       
    ##  3rd Qu.: 13.00   3rd Qu.: 4.000   3rd Qu.:  3.000   3rd Qu.: 1.00   3rd Qu.:4.866       
    ##  Max.   :140.00   Max.   :56.000   Max.   :100.000   Max.   :74.00   Max.   :6.198       
    ##   num_keywords    data_channel_is_lifestyle data_channel_is_entertainment data_channel_is_bus
    ##  Min.   : 1.000   Min.   :0.00000           Min.   :0.0000                Min.   :0.0000     
    ##  1st Qu.: 6.000   1st Qu.:0.00000           1st Qu.:0.0000                1st Qu.:0.0000     
    ##  Median : 7.000   Median :0.00000           Median :0.0000                Median :0.0000     
    ##  Mean   : 7.156   Mean   :0.05112           Mean   :0.1697                Mean   :0.1764     
    ##  3rd Qu.: 9.000   3rd Qu.:0.00000           3rd Qu.:0.0000                3rd Qu.:0.0000     
    ##  Max.   :10.000   Max.   :1.00000           Max.   :1.0000                Max.   :1.0000     
    ##  data_channel_is_socmed data_channel_is_tech data_channel_is_world   kw_min_min    
    ##  Min.   :0.00000        Min.   :0.0000       Min.   :0.0000        Min.   : -1.00  
    ##  1st Qu.:0.00000        1st Qu.:0.0000       1st Qu.:0.0000        1st Qu.: -1.00  
    ##  Median :0.00000        Median :0.0000       Median :0.0000        Median : -1.00  
    ##  Mean   :0.06076        Mean   :0.1754       Mean   :0.2177        Mean   : 27.39  
    ##  3rd Qu.:0.00000        3rd Qu.:0.0000       3rd Qu.:0.0000        3rd Qu.:  4.00  
    ##  Max.   :1.00000        Max.   :1.0000       Max.   :1.0000        Max.   :377.00  
    ##    kw_max_min       kw_avg_min        kw_min_max       kw_max_max       kw_avg_max    
    ##  Min.   :     0   Min.   :   -1.0   Min.   :     0   Min.   : 11100   Min.   :  3120  
    ##  1st Qu.:   451   1st Qu.:  143.8   1st Qu.:     0   1st Qu.:843300   1st Qu.:173769  
    ##  Median :   651   Median :  237.2   Median :  1400   Median :843300   Median :246802  
    ##  Mean   :  1178   Mean   :  317.4   Mean   : 14163   Mean   :749665   Mean   :262204  
    ##  3rd Qu.:  1000   3rd Qu.:  355.9   3rd Qu.:  7800   3rd Qu.:843300   3rd Qu.:336051  
    ##  Max.   :128500   Max.   :21516.0   Max.   :843300   Max.   :843300   Max.   :843300  
    ##    kw_min_avg       kw_max_avg       kw_avg_avg    self_reference_min_shares
    ##  Min.   :   0.0   Min.   :  2241   Min.   :  489   Min.   :     0           
    ##  1st Qu.:   0.0   1st Qu.:  3573   1st Qu.: 2381   1st Qu.:   600           
    ##  Median : 998.2   Median :  4333   Median : 2865   Median :  1200           
    ##  Mean   :1098.5   Mean   :  5665   Mean   : 3126   Mean   :  3831           
    ##  3rd Qu.:2026.0   3rd Qu.:  6012   3rd Qu.: 3569   3rd Qu.:  2500           
    ##  Max.   :3610.1   Max.   :128500   Max.   :24260   Max.   :690400           
    ##  self_reference_max_shares self_reference_avg_sharess   is_weekend     LDA_00       
    ##  Min.   :     0.0          Min.   :     0             Min.   :0    Min.   :0.01818  
    ##  1st Qu.:   987.8          1st Qu.:   926             1st Qu.:0    1st Qu.:0.02526  
    ##  Median :  2700.0          Median :  2185             Median :0    Median :0.03430  
    ##  Mean   :  9813.3          Mean   :  6162             Mean   :0    Mean   :0.19720  
    ##  3rd Qu.:  7700.0          3rd Qu.:  5040             3rd Qu.:0    3rd Qu.:0.28476  
    ##  Max.   :690400.0          Max.   :690400             Max.   :0    Max.   :0.92000  
    ##      LDA_01            LDA_02            LDA_03            LDA_04        global_subjectivity
    ##  Min.   :0.01818   Min.   :0.01818   Min.   :0.01818   Min.   :0.01818   Min.   :0.0000     
    ##  1st Qu.:0.02502   1st Qu.:0.02857   1st Qu.:0.02655   1st Qu.:0.02857   1st Qu.:0.3957     
    ##  Median :0.03335   Median :0.04002   Median :0.04000   Median :0.05000   Median :0.4523     
    ##  Mean   :0.13972   Mean   :0.21750   Mean   :0.21549   Mean   :0.23009   Mean   :0.4423     
    ##  3rd Qu.:0.15178   3rd Qu.:0.32720   3rd Qu.:0.35064   3rd Qu.:0.39236   3rd Qu.:0.5082     
    ##  Max.   :0.91997   Max.   :0.92000   Max.   :0.91994   Max.   :0.92645   Max.   :0.9222     
    ##  global_sentiment_polarity global_rate_positive_words global_rate_negative_words
    ##  Min.   :-0.37766          Min.   :0.00000            Min.   :0.000000          
    ##  1st Qu.: 0.05724          1st Qu.:0.02862            1st Qu.:0.009529          
    ##  Median : 0.11854          Median :0.03910            Median :0.015469          
    ##  Mean   : 0.11899          Mean   :0.03953            Mean   :0.016609          
    ##  3rd Qu.: 0.17803          3rd Qu.:0.05034            3rd Qu.:0.021602          
    ##  Max.   : 0.72784          Max.   :0.15278            Max.   :0.101695          
    ##  rate_positive_words rate_negative_words avg_positive_polarity min_positive_polarity
    ##  Min.   :0.0000      Min.   :0.0000      Min.   :0.0000        Min.   :0.00000      
    ##  1st Qu.:0.6000      1st Qu.:0.1818      1st Qu.:0.3048        1st Qu.:0.05000      
    ##  Median :0.7122      Median :0.2800      Median :0.3580        Median :0.10000      
    ##  Mean   :0.6817      Mean   :0.2882      Mean   :0.3519        Mean   :0.09614      
    ##  3rd Qu.:0.8000      3rd Qu.:0.3813      3rd Qu.:0.4119        3rd Qu.:0.10000      
    ##  Max.   :1.0000      Max.   :1.0000      Max.   :0.8500        Max.   :0.75000      
    ##  max_positive_polarity avg_negative_polarity min_negative_polarity max_negative_polarity
    ##  Min.   :0.0000        Min.   :-1.0000       Min.   :-1.0000       Min.   :-1.0000      
    ##  1st Qu.:0.6000        1st Qu.:-0.3250       1st Qu.:-0.7000       1st Qu.:-0.1250      
    ##  Median :0.8000        Median :-0.2500       Median :-0.5000       Median :-0.1000      
    ##  Mean   :0.7516        Mean   :-0.2569       Mean   :-0.5191       Mean   :-0.1072      
    ##  3rd Qu.:1.0000        3rd Qu.:-0.1833       3rd Qu.:-0.3000       3rd Qu.:-0.0500      
    ##  Max.   :1.0000        Max.   : 0.0000       Max.   : 0.0000       Max.   : 0.0000      
    ##  title_subjectivity title_sentiment_polarity abs_title_subjectivity
    ##  Min.   :0.0000     Min.   :-1.00000         Min.   :0.0000        
    ##  1st Qu.:0.0000     1st Qu.: 0.00000         1st Qu.:0.1667        
    ##  Median :0.1667     Median : 0.00000         Median :0.5000        
    ##  Mean   :0.2874     Mean   : 0.06867         Mean   :0.3429        
    ##  3rd Qu.:0.5000     3rd Qu.: 0.13854         3rd Qu.:0.5000        
    ##  Max.   :1.0000     Max.   : 1.00000         Max.   :0.5000        
    ##  abs_title_sentiment_polarity     shares      
    ##  Min.   :0.00000              Min.   :     8  
    ##  1st Qu.:0.00000              1st Qu.:   901  
    ##  Median :0.01151              Median :  1400  
    ##  Mean   :0.15413              Mean   :  3188  
    ##  3rd Qu.:0.25000              3rd Qu.:  2600  
    ##  Max.   :1.00000              Max.   :298400

``` r
correlation1 <- cor(train_data[,c(1:10,52)])
corrplot(correlation1,type='upper',tl.pos = 'lt')
corrplot(correlation1,type='lower',method = 'number',add = T,diag = F,tl.pos = 'n')
```

![](weekday_is_thursday_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

``` r
correlation2 <- cor(train_data[,c(11:20,52)])
corrplot(correlation2,type='upper',tl.pos = 'lt')
corrplot(correlation2,type='lower',method = 'number',add = T,diag = F,tl.pos = 'n')
```

![](weekday_is_thursday_files/figure-gfm/unnamed-chunk-6-2.png)<!-- -->

``` r
correlation3 <- cor(train_data[,c(21:30,52)])
```

    ## Warning in cor(train_data[, c(21:30, 52)]): the standard deviation is zero

``` r
corrplot(correlation3,type='upper',tl.pos = 'lt')
corrplot(correlation3,type='lower',method = 'number',add = T,diag = F,tl.pos = 'n')
```

![](weekday_is_thursday_files/figure-gfm/unnamed-chunk-6-3.png)<!-- -->

``` r
correlation4 <- cor(train_data[,c(31:40,52)])
corrplot(correlation4,type='upper',tl.pos = 'lt')
corrplot(correlation4,type='lower',method = 'number',add = T,diag = F,tl.pos = 'n')
```

![](weekday_is_thursday_files/figure-gfm/unnamed-chunk-6-4.png)<!-- -->

``` r
correlation5 <- cor(train_data[,c(41:51,52)])
corrplot(correlation5,type='upper',tl.pos = 'lt')
corrplot(correlation5,type='lower',method = 'number',add = T,diag = F,tl.pos = 'n')
```

![](weekday_is_thursday_files/figure-gfm/unnamed-chunk-6-5.png)<!-- -->

Unfortunately I did not find any variables are strongly related with the
response,so my plan is remove some highly correlated predictive
variables. From the correlation plot,I decided to remove some
meaningless variables:`is_weekend`,variables start with “LDA”. Also some
highly correlated variables will be removed too,like variables start
with“kw”,then we will get a new train set and test set.

``` r
#Remove meaningless variables
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

## Compare RMSE

We will make predictions using beset model fits and test set to compare
the RMSE of the two models.We will choose the model with a smaller RMSE
as our optimal model.

``` r
# predict values on test set and compare RMSE for two models
pred.tree <- predict(tree.method,test_data)
pred.boost <- predict(boosted.method,test_data)
compare <- rbind(postResample(pred.tree,test_data$shares),postResample(pred.boost,test_data$shares))
rownames(compare)<-c("Tree method","Boosted method")
compare
```

    ##                    RMSE   Rsquared      MAE
    ## Tree method    9386.627         NA 3004.182
    ## Boosted method 9189.113 0.04578476 2922.425

We generates two very similar RMSE,the smaller is preferred. In this
case,we can see the boosted method generates smaller RMSE which is the
same as we expected.The boosted method tend to have a better prediction
than the tree based method.
