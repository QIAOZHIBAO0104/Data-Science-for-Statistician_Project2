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

    ## [1] "weekday_is_friday"

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
    ##  1st Qu.: 9.00   1st Qu.: 235.0   1st Qu.:0.4763   1st Qu.:1.0000   1st Qu.:0.6311          
    ##  Median :10.00   Median : 408.0   Median :0.5456   Median :1.0000   Median :0.6964          
    ##  Mean   :10.42   Mean   : 529.9   Mean   :0.5375   Mean   :0.9714   Mean   :0.6805          
    ##  3rd Qu.:12.00   3rd Qu.: 698.8   3rd Qu.:0.6168   3rd Qu.:1.0000   3rd Qu.:0.7627          
    ##  Max.   :23.00   Max.   :7413.0   Max.   :0.9474   Max.   :1.0000   Max.   :1.0000          
    ##    num_hrefs      num_self_hrefs      num_imgs         num_videos     average_token_length
    ##  Min.   :  0.00   Min.   : 0.000   Min.   :  0.000   Min.   : 0.000   Min.   :0.000       
    ##  1st Qu.:  4.00   1st Qu.: 1.000   1st Qu.:  1.000   1st Qu.: 0.000   1st Qu.:4.470       
    ##  Median :  7.00   Median : 2.000   Median :  1.000   Median : 0.000   Median :4.663       
    ##  Mean   : 10.71   Mean   : 3.049   Mean   :  4.434   Mean   : 1.314   Mean   :4.555       
    ##  3rd Qu.: 13.00   3rd Qu.: 4.000   3rd Qu.:  3.000   3rd Qu.: 1.000   3rd Qu.:4.862       
    ##  Max.   :186.00   Max.   :51.000   Max.   :108.000   Max.   :91.000   Max.   :6.431       
    ##   num_keywords    data_channel_is_lifestyle data_channel_is_entertainment data_channel_is_bus
    ##  Min.   : 1.000   Min.   :0.00000           Min.   :0.0000                Min.   :0.0000     
    ##  1st Qu.: 6.000   1st Qu.:0.00000           1st Qu.:0.0000                1st Qu.:0.0000     
    ##  Median : 7.000   Median :0.00000           Median :0.0000                Median :0.0000     
    ##  Mean   : 7.217   Mean   :0.05263           Mean   :0.1714                Mean   :0.1534     
    ##  3rd Qu.: 9.000   3rd Qu.:0.00000           3rd Qu.:0.0000                3rd Qu.:0.0000     
    ##  Max.   :10.000   Max.   :1.00000           Max.   :1.0000                Max.   :1.0000     
    ##  data_channel_is_socmed data_channel_is_tech data_channel_is_world   kw_min_min       kw_max_min    
    ##  Min.   :0.00000        Min.   :0.0000       Min.   :0.0000        Min.   : -1.00   Min.   :     0  
    ##  1st Qu.:0.00000        1st Qu.:0.0000       1st Qu.:0.0000        1st Qu.: -1.00   1st Qu.:   451  
    ##  Median :0.00000        Median :0.0000       Median :0.0000        Median : -1.00   Median :   669  
    ##  Mean   :0.05539        Mean   :0.1687       Mean   :0.2273        Mean   : 26.99   Mean   :  1116  
    ##  3rd Qu.:0.00000        3rd Qu.:0.0000       3rd Qu.:0.0000        3rd Qu.:  4.00   3rd Qu.:  1000  
    ##  Max.   :1.00000        Max.   :1.0000       Max.   :1.0000        Max.   :217.00   Max.   :158900  
    ##    kw_avg_min        kw_min_max       kw_max_max       kw_avg_max       kw_min_avg     kw_max_avg    
    ##  Min.   :   -1.0   Min.   :     0   Min.   : 28000   Min.   :  5362   Min.   :  -1   Min.   :  2195  
    ##  1st Qu.:  142.4   1st Qu.:     0   1st Qu.:843300   1st Qu.:174738   1st Qu.:   0   1st Qu.:  3570  
    ##  Median :  235.8   Median :  1500   Median :843300   Median :247196   Median :1057   Median :  4398  
    ##  Mean   :  312.4   Mean   : 12478   Mean   :753092   Mean   :261058   Mean   :1113   Mean   :  5662  
    ##  3rd Qu.:  353.5   3rd Qu.:  7475   3rd Qu.:843300   3rd Qu.:334226   3rd Qu.:2007   3rd Qu.:  6079  
    ##  Max.   :39979.0   Max.   :843300   Max.   :843300   Max.   :843300   Max.   :3609   Max.   :171030  
    ##    kw_avg_avg      self_reference_min_shares self_reference_max_shares self_reference_avg_sharess
    ##  Min.   :  776.1   Min.   :     0.0          Min.   :     0            Min.   :     0.0          
    ##  1st Qu.: 2381.3   1st Qu.:   637.2          1st Qu.:  1000            1st Qu.:   944.5          
    ##  Median : 2858.1   Median :  1200.0          Median :  2800            Median :  2200.0          
    ##  Mean   : 3152.7   Mean   :  3945.0          Mean   : 10656            Mean   :  6501.1          
    ##  3rd Qu.: 3617.0   3rd Qu.:  2700.0          3rd Qu.:  7700            3rd Qu.:  5100.0          
    ##  Max.   :37607.5   Max.   :663600.0          Max.   :843300            Max.   :663600.0          
    ##    is_weekend     LDA_00            LDA_01            LDA_02            LDA_03            LDA_04       
    ##  Min.   :0    Min.   :0.01818   Min.   :0.01818   Min.   :0.01818   Min.   :0.01818   Min.   :0.01819  
    ##  1st Qu.:0    1st Qu.:0.02503   1st Qu.:0.02502   1st Qu.:0.02857   1st Qu.:0.02655   1st Qu.:0.02857  
    ##  Median :0    Median :0.03335   Median :0.03334   Median :0.04003   Median :0.04000   Median :0.04003  
    ##  Mean   :0    Mean   :0.17576   Mean   :0.13937   Mean   :0.22754   Mean   :0.23209   Mean   :0.22524  
    ##  3rd Qu.:0    3rd Qu.:0.23156   3rd Qu.:0.15038   3rd Qu.:0.36692   3rd Qu.:0.39842   3rd Qu.:0.37212  
    ##  Max.   :0    Max.   :0.92000   Max.   :0.91998   Max.   :0.92000   Max.   :0.92554   Max.   :0.92653  
    ##  global_subjectivity global_sentiment_polarity global_rate_positive_words global_rate_negative_words
    ##  Min.   :0.0000      Min.   :-0.36425          Min.   :0.00000            Min.   :0.00000           
    ##  1st Qu.:0.3987      1st Qu.: 0.05357          1st Qu.:0.02792            1st Qu.:0.01000           
    ##  Median :0.4560      Median : 0.11401          Median :0.03857            Median :0.01550           
    ##  Mean   :0.4467      Mean   : 0.11553          Mean   :0.03919            Mean   :0.01705           
    ##  3rd Qu.:0.5123      3rd Qu.: 0.17400          3rd Qu.:0.04982            3rd Qu.:0.02221           
    ##  Max.   :0.9500      Max.   : 0.61389          Max.   :0.13699            Max.   :0.13693           
    ##  rate_positive_words rate_negative_words avg_positive_polarity min_positive_polarity
    ##  Min.   :0.0000      Min.   :0.0000      Min.   :0.0000        Min.   :0.00000      
    ##  1st Qu.:0.5940      1st Qu.:0.2000      1st Qu.:0.3068        1st Qu.:0.05000      
    ##  Median :0.7000      Median :0.2857      Median :0.3584        Median :0.10000      
    ##  Mean   :0.6755      Mean   :0.2957      Mean   :0.3547        Mean   :0.09757      
    ##  3rd Qu.:0.7938      3rd Qu.:0.3902      3rd Qu.:0.4119        3rd Qu.:0.10000      
    ##  Max.   :1.0000      Max.   :1.0000      Max.   :1.0000        Max.   :1.00000      
    ##  max_positive_polarity avg_negative_polarity min_negative_polarity max_negative_polarity
    ##  Min.   :0.0000        Min.   :-1.0000       Min.   :-1.0000       Min.   :-1.0000      
    ##  1st Qu.:0.6000        1st Qu.:-0.3324       1st Qu.:-0.7000       1st Qu.:-0.1250      
    ##  Median :0.8000        Median :-0.2600       Median :-0.5000       Median :-0.1000      
    ##  Mean   :0.7522        Mean   :-0.2642       Mean   :-0.5236       Mean   :-0.1115      
    ##  3rd Qu.:1.0000        3rd Qu.:-0.1889       3rd Qu.:-0.3000       3rd Qu.:-0.0500      
    ##  Max.   :1.0000        Max.   : 0.0000       Max.   : 0.0000       Max.   : 0.0000      
    ##  title_subjectivity title_sentiment_polarity abs_title_subjectivity abs_title_sentiment_polarity
    ##  Min.   :0.0000     Min.   :-1.00000         Min.   :0.0000         Min.   :0.0000              
    ##  1st Qu.:0.0000     1st Qu.: 0.00000         1st Qu.:0.1667         1st Qu.:0.0000              
    ##  Median :0.1000     Median : 0.00000         Median :0.5000         Median :0.0000              
    ##  Mean   :0.2781     Mean   : 0.06749         Mean   :0.3473         Mean   :0.1512              
    ##  3rd Qu.:0.5000     3rd Qu.: 0.13636         3rd Qu.:0.5000         3rd Qu.:0.2500              
    ##  Max.   :1.0000     Max.   : 1.00000         Max.   :0.5000         Max.   :1.0000              
    ##      shares        
    ##  Min.   :    22.0  
    ##  1st Qu.:   984.2  
    ##  Median :  1500.0  
    ##  Mean   :  3309.2  
    ##  3rd Qu.:  2700.0  
    ##  Max.   :233400.0

``` r
correlation1 <- cor(train_data[,c(1:10,52)])
corrplot(correlation1,type='upper',tl.pos = 'lt')
corrplot(correlation1,type='lower',method = 'number',add = T,diag = F,tl.pos = 'n')
```

![](weekday_is_friday_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

``` r
correlation2 <- cor(train_data[,c(11:20,52)])
corrplot(correlation2,type='upper',tl.pos = 'lt')
corrplot(correlation2,type='lower',method = 'number',add = T,diag = F,tl.pos = 'n')
```

![](weekday_is_friday_files/figure-gfm/unnamed-chunk-6-2.png)<!-- -->

``` r
correlation3 <- cor(train_data[,c(21:30,52)])
```

    ## Warning in cor(train_data[, c(21:30, 52)]): the standard deviation is zero

``` r
corrplot(correlation3,type='upper',tl.pos = 'lt')
corrplot(correlation3,type='lower',method = 'number',add = T,diag = F,tl.pos = 'n')
```

![](weekday_is_friday_files/figure-gfm/unnamed-chunk-6-3.png)<!-- -->

``` r
correlation4 <- cor(train_data[,c(31:40,52)])
corrplot(correlation4,type='upper',tl.pos = 'lt')
corrplot(correlation4,type='lower',method = 'number',add = T,diag = F,tl.pos = 'n')
```

![](weekday_is_friday_files/figure-gfm/unnamed-chunk-6-4.png)<!-- -->

``` r
correlation5 <- cor(train_data[,c(41:51,52)])
corrplot(correlation5,type='upper',tl.pos = 'lt')
corrplot(correlation5,type='lower',method = 'number',add = T,diag = F,tl.pos = 'n')
```

![](weekday_is_friday_files/figure-gfm/unnamed-chunk-6-5.png)<!-- -->

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

    ##                         1        2        3       4        5        6
    ## Predicted Shares 2996.122 1969.458 5518.457 1965.87 2255.311 5540.074
    ## Actual Shares    7600.000 1500.000  998.000  885.00 1300.000 1300.000

``` r
# Calculate test RMSE
rmse.test.lynn <- sqrt(mean((pred.lynn - test_data$shares)^2))
```

As we can see, the naively produced and trained linear model is terrible
at prediction, with a huge RMSE of 7907.1957313 and tiny R-squared
0.0329198 that both reflect poor fit. The mini-comparison table between
the model predictions and actual values shows a huge discrepancy. The
test RMSE was calculated to be 8351.802787.
