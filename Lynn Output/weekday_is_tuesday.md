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

    ## [1] "weekday_is_tuesday"

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

    ##  n_tokens_title  n_tokens_content n_unique_tokens    n_non_stop_words   n_non_stop_unique_tokens
    ##  Min.   : 4.00   Min.   :   0.0   Min.   :  0.0000   Min.   :   0.000   Min.   :  0.0000        
    ##  1st Qu.: 9.00   1st Qu.: 246.0   1st Qu.:  0.4723   1st Qu.:   1.000   1st Qu.:  0.6293        
    ##  Median :10.00   Median : 398.0   Median :  0.5412   Median :   1.000   Median :  0.6914        
    ##  Mean   :10.46   Mean   : 543.4   Mean   :  0.6662   Mean   :   1.171   Mean   :  0.7984        
    ##  3rd Qu.:12.00   3rd Qu.: 689.0   3rd Qu.:  0.6098   3rd Qu.:   1.000   3rd Qu.:  0.7553        
    ##  Max.   :19.00   Max.   :7081.0   Max.   :701.0000   Max.   :1042.000   Max.   :650.0000        
    ##    num_hrefs      num_self_hrefs      num_imgs         num_videos    average_token_length
    ##  Min.   :  0.00   Min.   : 0.000   Min.   :  0.000   Min.   : 0.00   Min.   :0.000       
    ##  1st Qu.:  4.00   1st Qu.: 1.000   1st Qu.:  1.000   1st Qu.: 0.00   1st Qu.:4.473       
    ##  Median :  7.00   Median : 3.000   Median :  1.000   Median : 0.00   Median :4.663       
    ##  Mean   : 10.65   Mean   : 3.325   Mean   :  4.457   Mean   : 1.33   Mean   :4.540       
    ##  3rd Qu.: 13.00   3rd Qu.: 4.000   3rd Qu.:  3.000   3rd Qu.: 1.00   3rd Qu.:4.853       
    ##  Max.   :304.00   Max.   :62.000   Max.   :100.000   Max.   :59.00   Max.   :7.975       
    ##   num_keywords    data_channel_is_lifestyle data_channel_is_entertainment data_channel_is_bus
    ##  Min.   : 1.000   Min.   :0.00000           Min.   :0.0000                Min.   :0.0000     
    ##  1st Qu.: 6.000   1st Qu.:0.00000           1st Qu.:0.0000                1st Qu.:0.0000     
    ##  Median : 7.000   Median :0.00000           Median :0.0000                Median :0.0000     
    ##  Mean   : 7.167   Mean   :0.04523           Mean   :0.1746                Mean   :0.1531     
    ##  3rd Qu.: 9.000   3rd Qu.:0.00000           3rd Qu.:0.0000                3rd Qu.:0.0000     
    ##  Max.   :10.000   Max.   :1.00000           Max.   :1.0000                Max.   :1.0000     
    ##  data_channel_is_socmed data_channel_is_tech data_channel_is_world   kw_min_min       kw_max_min    
    ##  Min.   :0.00000        Min.   :0.0000       Min.   :0.0000        Min.   : -1.00   Min.   :     0  
    ##  1st Qu.:0.00000        1st Qu.:0.0000       1st Qu.:0.0000        1st Qu.: -1.00   1st Qu.:   440  
    ##  Median :0.00000        Median :0.0000       Median :0.0000        Median : -1.00   Median :   656  
    ##  Mean   :0.06418        Mean   :0.2014       Mean   :0.2138        Mean   : 25.02   Mean   :  1131  
    ##  3rd Qu.:0.00000        3rd Qu.:0.0000       3rd Qu.:0.0000        3rd Qu.:  4.00   3rd Qu.:  1000  
    ##  Max.   :1.00000        Max.   :1.0000       Max.   :1.0000        Max.   :217.00   Max.   :139600  
    ##    kw_avg_min        kw_min_max       kw_max_max       kw_avg_max       kw_min_avg       kw_max_avg    
    ##  Min.   :   -1.0   Min.   :     0   Min.   : 17100   Min.   :  3617   Min.   :   0.0   Min.   :  2019  
    ##  1st Qu.:  139.1   1st Qu.:     0   1st Qu.:843300   1st Qu.:171550   1st Qu.:   0.0   1st Qu.:  3529  
    ##  Median :  232.2   Median :  1300   Median :843300   Median :243857   Median : 991.6   Median :  4286  
    ##  Mean   :  307.7   Mean   : 13975   Mean   :755769   Mean   :262216   Mean   :1109.4   Mean   :  5619  
    ##  3rd Qu.:  357.0   3rd Qu.:  8600   3rd Qu.:843300   3rd Qu.:335554   3rd Qu.:2062.6   3rd Qu.:  6020  
    ##  Max.   :15851.2   Max.   :843300   Max.   :843300   Max.   :843300   Max.   :3609.7   Max.   :178675  
    ##    kw_avg_avg      self_reference_min_shares self_reference_max_shares self_reference_avg_sharess
    ##  Min.   :  804.4   Min.   :     0            Min.   :     0            Min.   :     0            
    ##  1st Qu.: 2368.3   1st Qu.:   636            1st Qu.:  1100            1st Qu.:   992            
    ##  Median : 2842.3   Median :  1200            Median :  2900            Median :  2250            
    ##  Mean   : 3129.8   Mean   :  4196            Mean   : 10221            Mean   :  6450            
    ##  3rd Qu.: 3575.9   3rd Qu.:  2700            3rd Qu.:  7900            3rd Qu.:  5200            
    ##  Max.   :29240.8   Max.   :690400            Max.   :843300            Max.   :690400            
    ##    is_weekend     LDA_00            LDA_01            LDA_02            LDA_03            LDA_04       
    ##  Min.   :0    Min.   :0.00000   Min.   :0.00000   Min.   :0.00000   Min.   :0.00000   Min.   :0.00000  
    ##  1st Qu.:0    1st Qu.:0.02506   1st Qu.:0.02501   1st Qu.:0.02857   1st Qu.:0.02857   1st Qu.:0.02858  
    ##  Median :0    Median :0.03337   Median :0.03334   Median :0.04004   Median :0.04000   Median :0.05000  
    ##  Mean   :0    Mean   :0.17780   Mean   :0.13432   Mean   :0.21945   Mean   :0.22107   Mean   :0.24718  
    ##  3rd Qu.:0    3rd Qu.:0.22811   3rd Qu.:0.13268   3rd Qu.:0.33900   3rd Qu.:0.35652   3rd Qu.:0.44016  
    ##  Max.   :0    Max.   :0.91998   Max.   :0.91994   Max.   :0.92000   Max.   :0.91997   Max.   :0.92719  
    ##  global_subjectivity global_sentiment_polarity global_rate_positive_words global_rate_negative_words
    ##  Min.   :0.0000      Min.   :-0.30881          Min.   :0.00000            Min.   :0.000000          
    ##  1st Qu.:0.3952      1st Qu.: 0.05682          1st Qu.:0.02829            1st Qu.:0.009317          
    ##  Median :0.4516      Median : 0.11964          Median :0.03881            Median :0.015038          
    ##  Mean   :0.4404      Mean   : 0.11855          Mean   :0.03933            Mean   :0.016257          
    ##  3rd Qu.:0.5052      3rd Qu.: 0.17639          3rd Qu.:0.04967            3rd Qu.:0.021244          
    ##  Max.   :0.8420      Max.   : 0.61923          Max.   :0.10714            Max.   :0.081352          
    ##  rate_positive_words rate_negative_words avg_positive_polarity min_positive_polarity
    ##  Min.   :0.0000      Min.   :0.0000      Min.   :0.0000        Min.   :0.00000      
    ##  1st Qu.:0.6000      1st Qu.:0.1842      1st Qu.:0.3045        1st Qu.:0.05000      
    ##  Median :0.7143      Median :0.2727      Median :0.3558        Median :0.10000      
    ##  Mean   :0.6843      Mean   :0.2845      Mean   :0.3496        Mean   :0.09483      
    ##  3rd Qu.:0.8000      3rd Qu.:0.3793      3rd Qu.:0.4069        3rd Qu.:0.10000      
    ##  Max.   :1.0000      Max.   :1.0000      Max.   :0.8333        Max.   :0.70000      
    ##  max_positive_polarity avg_negative_polarity min_negative_polarity max_negative_polarity
    ##  Min.   :0.0000        Min.   :-1.0000       Min.   :-1.0000       Min.   :-1.0000      
    ##  1st Qu.:0.6000        1st Qu.:-0.3250       1st Qu.:-0.7000       1st Qu.:-0.1250      
    ##  Median :0.8000        Median :-0.2500       Median :-0.5000       Median :-0.1000      
    ##  Mean   :0.7492        Mean   :-0.2557       Mean   :-0.5144       Mean   :-0.1069      
    ##  3rd Qu.:1.0000        3rd Qu.:-0.1826       3rd Qu.:-0.3000       3rd Qu.:-0.0500      
    ##  Max.   :1.0000        Max.   : 0.0000       Max.   : 0.0000       Max.   : 0.0000      
    ##  title_subjectivity title_sentiment_polarity abs_title_subjectivity abs_title_sentiment_polarity
    ##  Min.   :0.000      Min.   :-1.00000         Min.   :0.0000         Min.   :0.0000              
    ##  1st Qu.:0.000      1st Qu.: 0.00000         1st Qu.:0.1667         1st Qu.:0.0000              
    ##  Median :0.100      Median : 0.00000         Median :0.5000         Median :0.0000              
    ##  Mean   :0.274      Mean   : 0.07309         Mean   :0.3478         Mean   :0.1505              
    ##  3rd Qu.:0.500      3rd Qu.: 0.13636         3rd Qu.:0.5000         3rd Qu.:0.2500              
    ##  Max.   :1.000      Max.   : 1.00000         Max.   :0.5000         Max.   :1.0000              
    ##      shares      
    ##  Min.   :    45  
    ##  1st Qu.:   893  
    ##  Median :  1300  
    ##  Mean   :  3185  
    ##  3rd Qu.:  2500  
    ##  Max.   :441000

``` r
correlation1 <- cor(train_data[,c(1:10,52)])
corrplot(correlation1,type='upper',tl.pos = 'lt')
corrplot(correlation1,type='lower',method = 'number',add = T,diag = F,tl.pos = 'n')
```

![](weekday_is_tuesday_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

``` r
correlation2 <- cor(train_data[,c(11:20,52)])
corrplot(correlation2,type='upper',tl.pos = 'lt')
corrplot(correlation2,type='lower',method = 'number',add = T,diag = F,tl.pos = 'n')
```

![](weekday_is_tuesday_files/figure-gfm/unnamed-chunk-6-2.png)<!-- -->

``` r
correlation3 <- cor(train_data[,c(21:30,52)])
```

    ## Warning in cor(train_data[, c(21:30, 52)]): the standard deviation is zero

``` r
corrplot(correlation3,type='upper',tl.pos = 'lt')
corrplot(correlation3,type='lower',method = 'number',add = T,diag = F,tl.pos = 'n')
```

![](weekday_is_tuesday_files/figure-gfm/unnamed-chunk-6-3.png)<!-- -->

``` r
correlation4 <- cor(train_data[,c(31:40,52)])
corrplot(correlation4,type='upper',tl.pos = 'lt')
corrplot(correlation4,type='lower',method = 'number',add = T,diag = F,tl.pos = 'n')
```

![](weekday_is_tuesday_files/figure-gfm/unnamed-chunk-6-4.png)<!-- -->

``` r
correlation5 <- cor(train_data[,c(41:51,52)])
corrplot(correlation5,type='upper',tl.pos = 'lt')
corrplot(correlation5,type='lower',method = 'number',add = T,diag = F,tl.pos = 'n')
```

![](weekday_is_tuesday_files/figure-gfm/unnamed-chunk-6-5.png)<!-- -->

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
    ## Predicted Shares 2086.521 2839.454 2703.066 2587.743 2998.254 3229.383
    ## Actual Shares    1700.000 2000.000 1100.000  751.000 1200.000 1300.000

``` r
# Calculate test RMSE
rmse.test.lynn <- sqrt(mean((pred.lynn - test_data$shares)^2))
```

As we can see, the naively produced and trained linear model is terrible
at prediction, with a huge RMSE of 8609.1303274 and tiny R-squared
0.0198274 that both reflect poor fit. The mini-comparison table between
the model predictions and actual values shows a huge discrepancy. The
test RMSE was calculated to be 9955.691127.
