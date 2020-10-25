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

    ## [1] "weekday_is_wednesday"

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
    ##  Min.   : 4.00   Min.   :   0.0   Min.   :0.0000   Min.   :0.0000   Min.   :0.0000          
    ##  1st Qu.: 9.00   1st Qu.: 243.0   1st Qu.:0.4707   1st Qu.:1.0000   1st Qu.:0.6277          
    ##  Median :10.00   Median : 400.0   Median :0.5415   Median :1.0000   Median :0.6947          
    ##  Mean   :10.43   Mean   : 528.1   Mean   :0.5322   Mean   :0.9685   Mean   :0.6752          
    ##  3rd Qu.:12.00   3rd Qu.: 696.0   3rd Qu.:0.6126   3rd Qu.:1.0000   3rd Qu.:0.7583          
    ##  Max.   :18.00   Max.   :4747.0   Max.   :0.9375   Max.   :1.0000   Max.   :1.0000          
    ##    num_hrefs      num_self_hrefs      num_imgs        num_videos     average_token_length
    ##  Min.   :  0.00   Min.   : 0.000   Min.   : 0.000   Min.   : 0.000   Min.   :0.000       
    ##  1st Qu.:  4.00   1st Qu.: 1.000   1st Qu.: 1.000   1st Qu.: 0.000   1st Qu.:4.476       
    ##  Median :  7.00   Median : 2.000   Median : 1.000   Median : 0.000   Median :4.665       
    ##  Mean   : 10.18   Mean   : 3.113   Mean   : 4.074   Mean   : 1.254   Mean   :4.539       
    ##  3rd Qu.: 13.00   3rd Qu.: 4.000   3rd Qu.: 3.000   3rd Qu.: 1.000   3rd Qu.:4.853       
    ##  Max.   :150.00   Max.   :41.000   Max.   :92.000   Max.   :73.000   Max.   :6.610       
    ##   num_keywords    data_channel_is_lifestyle data_channel_is_entertainment data_channel_is_bus
    ##  Min.   : 1.000   Min.   :0.00000           Min.   :0.0000                Min.   :0.0000     
    ##  1st Qu.: 6.000   1st Qu.:0.00000           1st Qu.:0.0000                1st Qu.:0.0000     
    ##  Median : 7.000   Median :0.00000           Median :0.0000                Median :0.0000     
    ##  Mean   : 7.131   Mean   :0.05265           Mean   :0.1733                Mean   :0.1693     
    ##  3rd Qu.: 9.000   3rd Qu.:0.00000           3rd Qu.:0.0000                3rd Qu.:0.0000     
    ##  Max.   :10.000   Max.   :1.00000           Max.   :1.0000                Max.   :1.0000     
    ##  data_channel_is_socmed data_channel_is_tech data_channel_is_world   kw_min_min       kw_max_min      
    ##  Min.   :0.00000        Min.   :0.0000       Min.   :0.0000        Min.   : -1.00   Min.   :     0.0  
    ##  1st Qu.:0.00000        1st Qu.:0.0000       1st Qu.:0.0000        1st Qu.: -1.00   1st Qu.:   444.0  
    ##  Median :0.00000        Median :0.0000       Median :0.0000        Median : -1.00   Median :   651.5  
    ##  Mean   :0.05496        Mean   :0.1845       Mean   :0.2158        Mean   : 27.51   Mean   :  1177.7  
    ##  3rd Qu.:0.00000        3rd Qu.:0.0000       3rd Qu.:0.0000        3rd Qu.:  4.00   3rd Qu.:  1000.0  
    ##  Max.   :1.00000        Max.   :1.0000       Max.   :1.0000        Max.   :294.00   Max.   :111300.0  
    ##    kw_avg_min        kw_min_max       kw_max_max       kw_avg_max       kw_min_avg     kw_max_avg    
    ##  Min.   :   -1.0   Min.   :     0   Min.   : 17100   Min.   :  2300   Min.   :  -1   Min.   :  1953  
    ##  1st Qu.:  138.6   1st Qu.:     0   1st Qu.:690400   1st Qu.:172978   1st Qu.:   0   1st Qu.:  3521  
    ##  Median :  236.0   Median :  1400   Median :843300   Median :246243   Median :1009   Median :  4257  
    ##  Mean   :  317.0   Mean   : 14825   Mean   :746551   Mean   :261913   Mean   :1101   Mean   :  5613  
    ##  3rd Qu.:  358.5   3rd Qu.:  7700   3rd Qu.:843300   3rd Qu.:335659   3rd Qu.:2023   3rd Qu.:  5937  
    ##  Max.   :18687.8   Max.   :843300   Max.   :843300   Max.   :843300   Max.   :3613   Max.   :112787  
    ##    kw_avg_avg      self_reference_min_shares self_reference_max_shares self_reference_avg_sharess
    ##  Min.   :  424.3   Min.   :     0.0          Min.   :     0.0          Min.   :     0.0          
    ##  1st Qu.: 2356.2   1st Qu.:   610.8          1st Qu.:   979.8          1st Qu.:   922.8          
    ##  Median : 2832.3   Median :  1200.0          Median :  2700.0          Median :  2150.0          
    ##  Mean   : 3109.5   Mean   :  3895.5          Mean   : 10260.8          Mean   :  6407.1          
    ##  3rd Qu.: 3534.7   3rd Qu.:  2700.0          3rd Qu.:  7825.0          3rd Qu.:  5100.0          
    ##  Max.   :21000.7   Max.   :690400.0          Max.   :837700.0          Max.   :690400.0          
    ##    is_weekend     LDA_00            LDA_01            LDA_02            LDA_03            LDA_04       
    ##  Min.   :0    Min.   :0.01828   Min.   :0.01819   Min.   :0.01819   Min.   :0.01820   Min.   :0.01818  
    ##  1st Qu.:0    1st Qu.:0.02512   1st Qu.:0.02504   1st Qu.:0.02857   1st Qu.:0.02857   1st Qu.:0.02858  
    ##  Median :0    Median :0.03348   Median :0.03335   Median :0.04003   Median :0.04000   Median :0.05000  
    ##  Mean   :0    Mean   :0.18844   Mean   :0.13753   Mean   :0.22033   Mean   :0.21979   Mean   :0.23392  
    ##  3rd Qu.:0    3rd Qu.:0.25223   3rd Qu.:0.14958   3rd Qu.:0.34746   3rd Qu.:0.36517   3rd Qu.:0.39822  
    ##  Max.   :0    Max.   :0.92000   Max.   :0.91998   Max.   :0.92000   Max.   :0.91998   Max.   :0.92712  
    ##  global_subjectivity global_sentiment_polarity global_rate_positive_words global_rate_negative_words
    ##  Min.   :0.0000      Min.   :-0.34167          Min.   :0.00000            Min.   :0.000000          
    ##  1st Qu.:0.3945      1st Qu.: 0.05762          1st Qu.:0.02789            1st Qu.:0.009479          
    ##  Median :0.4528      Median : 0.11956          Median :0.03854            Median :0.014976          
    ##  Mean   :0.4413      Mean   : 0.11826          Mean   :0.03924            Mean   :0.016270          
    ##  3rd Qu.:0.5047      3rd Qu.: 0.17606          3rd Qu.:0.04973            3rd Qu.:0.021408          
    ##  Max.   :1.0000      Max.   : 0.57374          Max.   :0.15549            Max.   :0.085897          
    ##  rate_positive_words rate_negative_words avg_positive_polarity min_positive_polarity
    ##  Min.   :0.0000      Min.   :0.0000      Min.   :0.0000        Min.   :0.00000      
    ##  1st Qu.:0.6000      1st Qu.:0.1818      1st Qu.:0.3047        1st Qu.:0.05000      
    ##  Median :0.7143      Median :0.2778      Median :0.3575        Median :0.10000      
    ##  Mean   :0.6828      Mean   :0.2857      Mean   :0.3514        Mean   :0.09571      
    ##  3rd Qu.:0.8000      3rd Qu.:0.3846      3rd Qu.:0.4080        3rd Qu.:0.10000      
    ##  Max.   :1.0000      Max.   :1.0000      Max.   :1.0000        Max.   :1.00000      
    ##  max_positive_polarity avg_negative_polarity min_negative_polarity max_negative_polarity
    ##  Min.   :0.0000        Min.   :-1.0000       Min.   :-1.0000       Min.   :-1.0000      
    ##  1st Qu.:0.6000        1st Qu.:-0.3271       1st Qu.:-0.7000       1st Qu.:-0.1250      
    ##  Median :0.8000        Median :-0.2500       Median :-0.5000       Median :-0.1000      
    ##  Mean   :0.7477        Mean   :-0.2561       Mean   :-0.5117       Mean   :-0.1065      
    ##  3rd Qu.:1.0000        3rd Qu.:-0.1826       3rd Qu.:-0.3000       3rd Qu.:-0.0500      
    ##  Max.   :1.0000        Max.   : 0.0000       Max.   : 0.0000       Max.   : 0.0000      
    ##  title_subjectivity title_sentiment_polarity abs_title_subjectivity abs_title_sentiment_polarity
    ##  Min.   :0.0000     Min.   :-1.0000          Min.   :0.0000         Min.   :0.0000              
    ##  1st Qu.:0.0000     1st Qu.: 0.0000          1st Qu.:0.1667         1st Qu.:0.0000              
    ##  Median :0.1000     Median : 0.0000          Median :0.5000         Median :0.0000              
    ##  Mean   :0.2764     Mean   : 0.0656          Mean   :0.3461         Mean   :0.1531              
    ##  3rd Qu.:0.5000     3rd Qu.: 0.1364          3rd Qu.:0.5000         3rd Qu.:0.2500              
    ##  Max.   :1.0000     Max.   : 1.0000          Max.   :0.5000         Max.   :1.0000              
    ##      shares      
    ##  Min.   :    36  
    ##  1st Qu.:   879  
    ##  Median :  1300  
    ##  Mean   :  3282  
    ##  3rd Qu.:  2600  
    ##  Max.   :843300

``` r
correlation1 <- cor(train_data[,c(1:10,52)])
corrplot(correlation1,type='upper',tl.pos = 'lt')
corrplot(correlation1,type='lower',method = 'number',add = T,diag = F,tl.pos = 'n')
```

![](weekday_is_wednesday_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

``` r
correlation2 <- cor(train_data[,c(11:20,52)])
corrplot(correlation2,type='upper',tl.pos = 'lt')
corrplot(correlation2,type='lower',method = 'number',add = T,diag = F,tl.pos = 'n')
```

![](weekday_is_wednesday_files/figure-gfm/unnamed-chunk-6-2.png)<!-- -->

``` r
correlation3 <- cor(train_data[,c(21:30,52)])
```

    ## Warning in cor(train_data[, c(21:30, 52)]): the standard deviation is zero

``` r
corrplot(correlation3,type='upper',tl.pos = 'lt')
corrplot(correlation3,type='lower',method = 'number',add = T,diag = F,tl.pos = 'n')
```

![](weekday_is_wednesday_files/figure-gfm/unnamed-chunk-6-3.png)<!-- -->

``` r
correlation4 <- cor(train_data[,c(31:40,52)])
corrplot(correlation4,type='upper',tl.pos = 'lt')
corrplot(correlation4,type='lower',method = 'number',add = T,diag = F,tl.pos = 'n')
```

![](weekday_is_wednesday_files/figure-gfm/unnamed-chunk-6-4.png)<!-- -->

``` r
correlation5 <- cor(train_data[,c(41:51,52)])
corrplot(correlation5,type='upper',tl.pos = 'lt')
corrplot(correlation5,type='lower',method = 'number',add = T,diag = F,tl.pos = 'n')
```

![](weekday_is_wednesday_files/figure-gfm/unnamed-chunk-6-5.png)<!-- -->

Unfortunately I did not find any variables are strongly related with the
response,so my plan is remove some highly correlated predictive
variables. From the correlation plot,I decided to remove some
meaningless variables:`is_weekend`,variables start with “LDA”. Also some
highly correlated variables will be removed too,like variables start
with“kw”,then we will get a new train set and test set.

``` r
#Remove meaningless variabls
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
boosted.method <- train(shares~.,data = train_data,method = 'gbm',
                      trControl = trainControl(method = 'repeatedcv', number=5,repeats =2),
                      preProcess = c("center","scale"),
                      verbose = FALSE)
boosted.method$results
boosted.method$bestTune
```

# Linear Model

Project partner Lynn Huang added a simple linear model here to practice
GitHub forking and pull requesting practices. Nothing fancy for the
model, which means an atrocious fit is to be expected\!

``` r
fit.lynn <- lm(shares ~ ., data=train_data)
summ <- summary(fit.lynn)
rsquared <- summ$adj.r.squared

# Let's only keep the significant predictors and do 10-fold CV on that
fit.lynnCV <- train(shares ~ num_hrefs + average_token_length + data_channel_is_lifestyle +
                      data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed +
                      data_channel_is_tech + data_channel_is_world + self_reference_min_shares,
                    data=train_data,
                    method="lm",
                    trControl=trainControl(method="cv", number=10))
# As expected, we have an atrocious fit (huge RMSE, tiny Rsquared)
fit.lynnCV$results
rmse.lynn <- fit.lynnCV$results$RMSE
rsquared.lynn <- fit.lynnCV$results$Rsquared

# Well, let's run this atrocious model on the test data!
pred.lynn <- predict(fit.lynnCV, newdata=test_data)
# A snapshot of the results
results.preview <- rbind(head(pred.lynn), head(test_data$shares))
rownames(results.preview) <- c("Predicted Shares", "Actual Shares")
results.preview
```

    ##                         1        2        3        4        5        6
    ## Predicted Shares 2834.362 1598.842 3741.708 1487.593 2580.119 1338.697
    ## Actual Shares    1100.000 1400.000  838.000 1100.000  533.000 1300.000

``` r
# Calculate test RMSE
rmse.test.lynn <- sqrt(mean((pred.lynn - test_data$shares)^2))
```

As we can see, the naively produced and trained linear model is terrible
at prediction, with a huge RMSE of 1.071148410^{4} and tiny R-squared
0.0303978 that both reflect poor fit. The mini-comparison table between
the model predictions and actual values shows a huge discrepancy. The
test RMSE was calculated to be 1.398383310^{4}.
