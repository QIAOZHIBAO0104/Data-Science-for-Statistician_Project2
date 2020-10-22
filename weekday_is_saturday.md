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

    ## [1] "weekday_is_saturday"

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

    ##  n_tokens_title  n_tokens_content n_unique_tokens  n_non_stop_words
    ##  Min.   : 5.00   Min.   :   0.0   Min.   :0.0000   Min.   :0.000   
    ##  1st Qu.: 9.00   1st Qu.: 277.0   1st Qu.:0.4595   1st Qu.:1.000   
    ##  Median :10.00   Median : 496.0   Median :0.5199   Median :1.000   
    ##  Mean   :10.24   Mean   : 596.8   Mean   :0.5112   Mean   :0.961   
    ##  3rd Qu.:12.00   3rd Qu.: 781.0   3rd Qu.:0.5927   3rd Qu.:1.000   
    ##  Max.   :18.00   Max.   :4046.0   Max.   :0.9574   Max.   :1.000   
    ##  n_non_stop_unique_tokens   num_hrefs      num_self_hrefs      num_imgs     
    ##  Min.   :0.0000           Min.   :  0.00   Min.   : 0.000   Min.   : 0.000  
    ##  1st Qu.:0.6116           1st Qu.:  5.00   1st Qu.: 1.000   1st Qu.: 1.000  
    ##  Median :0.6742           Median : 10.00   Median : 3.000   Median : 1.000  
    ##  Mean   :0.6530           Mean   : 13.03   Mean   : 3.917   Mean   : 5.411  
    ##  3rd Qu.:0.7361           3rd Qu.: 17.00   3rd Qu.: 4.000   3rd Qu.: 8.000  
    ##  Max.   :1.0000           Max.   :105.00   Max.   :74.000   Max.   :99.000  
    ##    num_videos     average_token_length  num_keywords    data_channel_is_lifestyle
    ##  Min.   : 0.000   Min.   :0.000        Min.   : 1.000   Min.   :0.0000           
    ##  1st Qu.: 0.000   1st Qu.:4.475        1st Qu.: 6.000   1st Qu.:0.0000           
    ##  Median : 0.000   Median :4.666        Median : 8.000   Median :0.0000           
    ##  Mean   : 1.161   Mean   :4.507        Mean   : 7.547   Mean   :0.0728           
    ##  3rd Qu.: 1.000   3rd Qu.:4.855        3rd Qu.: 9.000   3rd Qu.:0.0000           
    ##  Max.   :74.000   Max.   :6.295        Max.   :10.000   Max.   :1.0000           
    ##  data_channel_is_entertainment data_channel_is_bus data_channel_is_socmed
    ##  Min.   :0.0000                Min.   :0.0000      Min.   :0.00000       
    ##  1st Qu.:0.0000                1st Qu.:0.0000      1st Qu.:0.00000       
    ##  Median :0.0000                Median :0.0000      Median :0.00000       
    ##  Mean   :0.1555                Mean   :0.1043      Mean   :0.06639       
    ##  3rd Qu.:0.0000                3rd Qu.:0.0000      3rd Qu.:0.00000       
    ##  Max.   :1.0000                Max.   :1.0000      Max.   :1.00000       
    ##  data_channel_is_tech data_channel_is_world   kw_min_min       kw_max_min   
    ##  Min.   :0.0000       Min.   :0.0000        Min.   : -1.00   Min.   :    0  
    ##  1st Qu.:0.0000       1st Qu.:0.0000        1st Qu.: -1.00   1st Qu.:  461  
    ##  Median :0.0000       Median :0.0000        Median : -1.00   Median :  695  
    ##  Mean   :0.2196       Mean   :0.2108        Mean   : 23.59   Mean   : 1096  
    ##  3rd Qu.:0.0000       3rd Qu.:0.0000        3rd Qu.:  4.00   3rd Qu.: 1100  
    ##  Max.   :1.0000       Max.   :1.0000        Max.   :217.00   Max.   :50100  
    ##    kw_avg_min       kw_min_max       kw_max_max       kw_avg_max       kw_min_avg  
    ##  Min.   :  -1.0   Min.   :     0   Min.   : 37400   Min.   :  7178   Min.   :   0  
    ##  1st Qu.: 142.8   1st Qu.:     0   1st Qu.:843300   1st Qu.:170944   1st Qu.:   0  
    ##  Median : 244.1   Median :  1900   Median :843300   Median :239888   Median :1266  
    ##  Mean   : 302.4   Mean   : 15659   Mean   :761791   Mean   :250369   Mean   :1260  
    ##  3rd Qu.: 362.9   3rd Qu.: 10800   3rd Qu.:843300   3rd Qu.:315360   3rd Qu.:2212  
    ##  Max.   :8549.3   Max.   :843300   Max.   :843300   Max.   :843300   Max.   :3594  
    ##    kw_max_avg       kw_avg_avg    self_reference_min_shares self_reference_max_shares
    ##  Min.   :  2414   Min.   : 1115   Min.   :     0            Min.   :     0           
    ##  1st Qu.:  3578   1st Qu.: 2505   1st Qu.:   681            1st Qu.:  1100           
    ##  Median :  4662   Median : 3045   Median :  1300            Median :  2900           
    ##  Mean   :  5982   Mean   : 3300   Mean   :  3753            Mean   : 10867           
    ##  3rd Qu.:  6620   3rd Qu.: 3839   3rd Qu.:  2700            3rd Qu.:  8200           
    ##  Max.   :237967   Max.   :36717   Max.   :663600            Max.   :837700           
    ##  self_reference_avg_sharess   is_weekend     LDA_00            LDA_01       
    ##  Min.   :     0             Min.   :1    Min.   :0.02000   Min.   :0.01819  
    ##  1st Qu.:  1000             1st Qu.:1    1st Qu.:0.02500   1st Qu.:0.02352  
    ##  Median :  2350             Median :1    Median :0.03333   Median :0.03333  
    ##  Mean   :  6087             Mean   :1    Mean   :0.16589   Mean   :0.13438  
    ##  3rd Qu.:  5200             3rd Qu.:1    3rd Qu.:0.19810   3rd Qu.:0.13412  
    ##  Max.   :663600             Max.   :1    Max.   :0.91998   Max.   :0.91996  
    ##      LDA_02            LDA_03            LDA_04        global_subjectivity
    ##  Min.   :0.01832   Min.   :0.01821   Min.   :0.02000   Min.   :0.0000     
    ##  1st Qu.:0.02500   1st Qu.:0.02502   1st Qu.:0.02857   1st Qu.:0.4063     
    ##  Median :0.04000   Median :0.04000   Median :0.05000   Median :0.4630     
    ##  Mean   :0.21386   Mean   :0.22585   Mean   :0.26003   Mean   :0.4491     
    ##  3rd Qu.:0.33416   3rd Qu.:0.38332   3rd Qu.:0.45662   3rd Qu.:0.5179     
    ##  Max.   :0.92000   Max.   :0.91997   Max.   :0.91999   Max.   :0.8125     
    ##  global_sentiment_polarity global_rate_positive_words global_rate_negative_words
    ##  Min.   :-0.2146           Min.   :0.00000            Min.   :0.000000          
    ##  1st Qu.: 0.0584           1st Qu.:0.02825            1st Qu.:0.009823          
    ##  Median : 0.1229           Median :0.04040            Median :0.015625          
    ##  Mean   : 0.1238           Mean   :0.04063            Mean   :0.016684          
    ##  3rd Qu.: 0.1893           3rd Qu.:0.05263            3rd Qu.:0.021906          
    ##  Max.   : 0.6000           Max.   :0.13065            Max.   :0.139831          
    ##  rate_positive_words rate_negative_words avg_positive_polarity min_positive_polarity
    ##  Min.   :0.0000      Min.   :0.0000      Min.   :0.0000        Min.   :0.00000      
    ##  1st Qu.:0.6000      1st Qu.:0.1818      1st Qu.:0.3126        1st Qu.:0.05000      
    ##  Median :0.7143      Median :0.2742      Median :0.3659        Median :0.10000      
    ##  Mean   :0.6778      Mean   :0.2832      Mean   :0.3567        Mean   :0.08906      
    ##  3rd Qu.:0.8000      3rd Qu.:0.3750      3rd Qu.:0.4167        3rd Qu.:0.10000      
    ##  Max.   :1.0000      Max.   :1.0000      Max.   :1.0000        Max.   :1.00000      
    ##  max_positive_polarity avg_negative_polarity min_negative_polarity max_negative_polarity
    ##  Min.   :0.0000        Min.   :-1.0000       Min.   :-1.0000       Min.   :-1.0000      
    ##  1st Qu.:0.6000        1st Qu.:-0.3396       1st Qu.:-0.8000       1st Qu.:-0.1250      
    ##  Median :0.8000        Median :-0.2607       Median :-0.5000       Median :-0.1000      
    ##  Mean   :0.7775        Mean   :-0.2668       Mean   :-0.5518       Mean   :-0.1037      
    ##  3rd Qu.:1.0000        3rd Qu.:-0.1988       3rd Qu.:-0.3889       3rd Qu.:-0.0500      
    ##  Max.   :1.0000        Max.   : 0.0000       Max.   : 0.0000       Max.   : 0.0000      
    ##  title_subjectivity title_sentiment_polarity abs_title_subjectivity
    ##  Min.   :0.0000     Min.   :-1.00000         Min.   :0.0000        
    ##  1st Qu.:0.0000     1st Qu.: 0.00000         1st Qu.:0.1250        
    ##  Median :0.2000     Median : 0.00000         Median :0.4500        
    ##  Mean   :0.2881     Mean   : 0.09693         Mean   :0.3276        
    ##  3rd Qu.:0.5000     3rd Qu.: 0.22500         3rd Qu.:0.5000        
    ##  Max.   :1.0000     Max.   : 1.00000         Max.   :0.5000        
    ##  abs_title_sentiment_polarity     shares      
    ##  Min.   :0.00000              Min.   :    49  
    ##  1st Qu.:0.00000              1st Qu.:  1400  
    ##  Median :0.03333              Median :  2000  
    ##  Mean   :0.16647              Mean   :  3806  
    ##  3rd Qu.:0.25000              3rd Qu.:  3700  
    ##  Max.   :1.00000              Max.   :144400

``` r
correlation1 <- cor(train_data[,c(1:10,52)])
corrplot(correlation1,type='upper',tl.pos = 'lt')
corrplot(correlation1,type='lower',method = 'number',add = T,diag = F,tl.pos = 'n')
```

![](weekday_is_saturday_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

``` r
correlation2 <- cor(train_data[,c(11:20,52)])
corrplot(correlation2,type='upper',tl.pos = 'lt')
corrplot(correlation2,type='lower',method = 'number',add = T,diag = F,tl.pos = 'n')
```

![](weekday_is_saturday_files/figure-gfm/unnamed-chunk-6-2.png)<!-- -->

``` r
correlation3 <- cor(train_data[,c(21:30,52)])
```

    ## Warning in cor(train_data[, c(21:30, 52)]): the standard deviation is zero

``` r
corrplot(correlation3,type='upper',tl.pos = 'lt')
corrplot(correlation3,type='lower',method = 'number',add = T,diag = F,tl.pos = 'n')
```

![](weekday_is_saturday_files/figure-gfm/unnamed-chunk-6-3.png)<!-- -->

``` r
correlation4 <- cor(train_data[,c(31:40,52)])
corrplot(correlation4,type='upper',tl.pos = 'lt')
corrplot(correlation4,type='lower',method = 'number',add = T,diag = F,tl.pos = 'n')
```

![](weekday_is_saturday_files/figure-gfm/unnamed-chunk-6-4.png)<!-- -->

``` r
correlation5 <- cor(train_data[,c(41:51,52)])
corrplot(correlation5,type='upper',tl.pos = 'lt')
corrplot(correlation5,type='lower',method = 'number',add = T,diag = F,tl.pos = 'n')
```

![](weekday_is_saturday_files/figure-gfm/unnamed-chunk-6-5.png)<!-- -->

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

# Second Analysis

## Linear model

As we already removed some predictor variables based on collinearity,to
simplify the variable selection we just pick some significant variables
from the linear fit model,then pick a model from the candidate models.

We picked `num_hrefs`, `tt{average_token_length`,
`data_channel_is_lifestyle`, `data_channel_is_entertainment`,
`data_channel_is_bus`,
`data_channel_is_socmed`,`data_channel_is_tech`,`data_channel_is_world`,`self_reference_min_shares`.

``` r
# fit a linear model
lm.fit <- lm(shares ~., data=train_data)
summary(lm.fit)
```

    ## 
    ## Call:
    ## lm(formula = shares ~ ., data = train_data)
    ## 
    ## Residuals:
    ##    Min     1Q Median     3Q    Max 
    ##  -8880  -2358  -1248    267 136033 
    ## 
    ## Coefficients: (1 not defined because of singularities)
    ##                                 Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                    1.584e+03  1.417e+03   1.118 0.263644    
    ## n_tokens_title                 1.538e+01  7.951e+01   0.193 0.846694    
    ## n_tokens_content               1.263e-02  6.544e-01   0.019 0.984605    
    ## n_unique_tokens                2.377e+03  5.817e+03   0.409 0.682831    
    ## n_non_stop_words              -5.208e+03  4.785e+03  -1.088 0.276551    
    ## n_non_stop_unique_tokens       2.148e+03  4.794e+03   0.448 0.654240    
    ## num_hrefs                      1.623e+01  1.792e+01   0.905 0.365429    
    ## num_self_hrefs                -4.718e+01  3.521e+01  -1.340 0.180422    
    ## num_imgs                       5.731e+00  2.270e+01   0.252 0.800700    
    ## num_videos                    -2.471e+01  4.234e+01  -0.583 0.559661    
    ## average_token_length           1.957e+02  7.242e+02   0.270 0.786997    
    ## num_keywords                   1.674e+02  8.785e+01   1.906 0.056872 .  
    ## data_channel_is_lifestyle     -1.565e+03  7.669e+02  -2.041 0.041387 *  
    ## data_channel_is_entertainment -1.761e+03  6.187e+02  -2.847 0.004469 ** 
    ## data_channel_is_bus           -4.926e+02  7.467e+02  -0.660 0.509493    
    ## data_channel_is_socmed        -1.005e+03  8.258e+02  -1.216 0.224029    
    ## data_channel_is_tech          -1.160e+03  6.211e+02  -1.868 0.061908 .  
    ## data_channel_is_world         -2.078e+03  6.558e+02  -3.168 0.001563 ** 
    ## self_reference_min_shares      1.501e-02  2.338e-02   0.642 0.521005    
    ## self_reference_max_shares      8.030e-03  9.858e-03   0.815 0.415437    
    ## self_reference_avg_sharess    -1.636e-02  3.024e-02  -0.541 0.588447    
    ## global_subjectivity            4.331e+03  2.506e+03   1.729 0.084082 .  
    ## global_sentiment_polarity     -5.052e+03  5.010e+03  -1.008 0.313435    
    ## global_rate_positive_words     9.188e+03  2.070e+04   0.444 0.657175    
    ## global_rate_negative_words    -7.272e+03  3.897e+04  -0.187 0.851978    
    ## rate_positive_words           -8.698e+02  3.240e+03  -0.268 0.788378    
    ## rate_negative_words                   NA         NA      NA       NA    
    ## avg_positive_polarity          5.861e+03  4.104e+03   1.428 0.153462    
    ## min_positive_polarity         -6.417e+03  3.430e+03  -1.871 0.061542 .  
    ## max_positive_polarity         -7.641e+02  1.246e+03  -0.613 0.539869    
    ## avg_negative_polarity          6.933e+02  3.433e+03   0.202 0.839987    
    ## min_negative_polarity          4.545e+02  1.225e+03   0.371 0.710668    
    ## max_negative_polarity         -6.068e+01  2.846e+03  -0.021 0.982993    
    ## title_subjectivity             5.175e+02  7.508e+02   0.689 0.490781    
    ## title_sentiment_polarity       5.656e+02  7.625e+02   0.742 0.458329    
    ## abs_title_subjectivity         3.962e+03  1.027e+03   3.860 0.000118 ***
    ## abs_title_sentiment_polarity   2.675e+03  1.122e+03   2.385 0.017174 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 6665 on 1681 degrees of freedom
    ## Multiple R-squared:  0.04635,    Adjusted R-squared:  0.0265 
    ## F-statistic: 2.334 on 35 and 1681 DF,  p-value: 1.887e-05

Then we will fit some candidate models to select.

``` r
# Candidate models
fit1 <- train(shares ~ num_hrefs, train_data,
             method='lm',
             trControl = trainControl(method = 'cv',number=10))
fit1$results$RMSE
```

    ## [1] 6377.911

``` r
fit2 <- train(shares ~ num_hrefs+average_token_length, train_data,
             method='lm',
             trControl = trainControl(method = 'cv',number=10))
fit2$results$RMSE
```

    ## [1] 6440.021

``` r
fit3 <- train(shares ~ num_hrefs+average_token_length
              +data_channel_is_lifestyle, train_data,
             method='lm',
             trControl = trainControl(method = 'cv',number=10))
fit3$results$RMSE
```

    ## [1] 6384.204

``` r
fit4 <- train(shares ~ num_hrefs+average_token_length
              +data_channel_is_lifestyle+data_channel_is_entertainment, train_data,
             method='lm',
             trControl = trainControl(method = 'cv',number=10))
fit4$results$RMSE
```

    ## [1] 6363.209

``` r
fit5 <- train(shares ~ num_hrefs+average_token_length
              +data_channel_is_lifestyle+data_channel_is_entertainment
              +data_channel_is_bus, train_data,
             method='lm',
             trControl = trainControl(method = 'cv',number=10))
fit5$results$RMSE
```

    ## [1] 6112.68

``` r
fit6 <- train(shares ~ num_hrefs+average_token_length
              +data_channel_is_lifestyle+data_channel_is_entertainment
              +data_channel_is_bus+data_channel_is_socmed, train_data,
             method='lm',
             trControl = trainControl(method = 'cv',number=10))
fit6$results$RMSE
```

    ## [1] 6247.674

``` r
fit7<- train(shares ~ num_hrefs+average_token_length
              +data_channel_is_lifestyle+data_channel_is_entertainment
              +data_channel_is_bus+data_channel_is_socmed+data_channel_is_tech, train_data,
             method='lm',
             trControl = trainControl(method = 'cv',number=10))
fit7$results$RMSE
```

    ## [1] 6301.488

``` r
fit8 <- train(shares ~ num_hrefs+average_token_length
              +data_channel_is_lifestyle+data_channel_is_entertainment
              +data_channel_is_bus+data_channel_is_socmed+data_channel_is_tech+
              data_channel_is_world,
              train_data,method='lm',
             trControl = trainControl(method = 'cv',number=10))
fit8$results$RMSE
```

    ## [1] 6106.706

``` r
fit9 <- train(shares ~ num_hrefs+average_token_length
              +data_channel_is_lifestyle+data_channel_is_entertainment
              +data_channel_is_bus+data_channel_is_socmed+data_channel_is_tech+
              data_channel_is_world+self_reference_min_shares,
              train_data,method='lm',
             trControl = trainControl(method = 'cv',number=10))
fit9$results$RMSE
```

    ## [1] 6187.837

Model fit7 has the lowest training RMSE with 7 variables, so we pick
this model as our optimal model for further evaluation on test set.

``` r
# final linear model
lm.fit.final <- as.formula(shares ~ num_hrefs+average_token_length
              +data_channel_is_lifestyle+data_channel_is_entertainment
              +data_channel_is_bus+data_channel_is_socmed+data_channel_is_tech)
```

Now we will calculate the test RMSE on test set for the picked linear
model.

``` r
final.fit <- train(as.formula(shares ~ num_hrefs+average_token_length
              +data_channel_is_lifestyle+data_channel_is_entertainment
              +data_channel_is_bus+data_channel_is_socmed+data_channel_is_tech),
              test_data,method='lm',
              trControl = trainControl(method = 'cv',number=5))
final.fit$results$RMSE
```

    ## [1] 6625.858

## Compare RMSE

We will make predictions using beset model fits and test set to compare
the RMSE of the three models.We will choose the model with a smaller
RMSE as our final optimal model.

``` r
# predict values on test set and compare RMSE for two models
pred.tree <- predict(tree.method,test_data)
pred.boost <- predict(boosted.method,test_data)
tree.rmse <- sqrt(mean((pred.tree-test_data$shares)^2))
boost.rmse <- sqrt(mean((pred.boost-test_data$shares)^2))
compare <- cbind(tree.rmse,boost.rmse,final.fit$results$RMSE)
colnames(compare)<-c("Tree method","Boosted method","Linear Model")
compare
```

    ##      Tree method Boosted method Linear Model
    ## [1,]    6752.639       6535.142     6625.858

In this case,we can see the linear model generates the smallest RMSE.The
linear model tend to have a better prediction than the tree based method
and boosted method.
