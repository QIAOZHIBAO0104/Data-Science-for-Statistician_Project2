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

    ## [1] "weekday_is_sunday"

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
    ##  Min.   : 3.00   Min.   :   0.0   Min.   :0.0000   Min.   :0.0000  
    ##  1st Qu.: 9.00   1st Qu.: 245.0   1st Qu.:0.4618   1st Qu.:1.0000  
    ##  Median :10.00   Median : 454.0   Median :0.5275   Median :1.0000  
    ##  Mean   :10.52   Mean   : 593.5   Mean   :0.5268   Mean   :0.9713  
    ##  3rd Qu.:12.00   3rd Qu.: 794.5   3rd Qu.:0.6073   3rd Qu.:1.0000  
    ##  Max.   :19.00   Max.   :8474.0   Max.   :0.9796   Max.   :1.0000  
    ##  n_non_stop_unique_tokens   num_hrefs      num_self_hrefs      num_imgs      
    ##  Min.   :0.0000           Min.   :  0.00   Min.   : 0.000   Min.   :  0.000  
    ##  1st Qu.:0.6128           1st Qu.:  5.00   1st Qu.: 1.000   1st Qu.:  1.000  
    ##  Median :0.6794           Median :  9.00   Median : 2.000   Median :  1.000  
    ##  Mean   :0.6649           Mean   : 12.52   Mean   : 3.556   Mean   :  5.897  
    ##  3rd Qu.:0.7524           3rd Qu.: 16.00   3rd Qu.: 4.000   3rd Qu.:  9.000  
    ##  Max.   :1.0000           Max.   :153.00   Max.   :39.000   Max.   :111.000  
    ##    num_videos      average_token_length  num_keywords    data_channel_is_lifestyle
    ##  Min.   : 0.0000   Min.   :0.000        Min.   : 1.000   Min.   :0.0000           
    ##  1st Qu.: 0.0000   1st Qu.:4.486        1st Qu.: 6.000   1st Qu.:0.0000           
    ##  Median : 0.0000   Median :4.683        Median : 8.000   Median :0.0000           
    ##  Mean   : 0.9843   Mean   :4.577        Mean   : 7.669   Mean   :0.0799           
    ##  3rd Qu.: 1.0000   3rd Qu.:4.873        3rd Qu.: 9.000   3rd Qu.:0.0000           
    ##  Max.   :73.0000   Max.   :5.994        Max.   :10.000   Max.   :1.0000           
    ##  data_channel_is_entertainment data_channel_is_bus data_channel_is_socmed
    ##  Min.   :0.0000                Min.   :0.0000      Min.   :0.00000       
    ##  1st Qu.:0.0000                1st Qu.:0.0000      1st Qu.:0.00000       
    ##  Median :0.0000                Median :0.0000      Median :0.00000       
    ##  Mean   :0.1906                Mean   :0.1243      Mean   :0.04648       
    ##  3rd Qu.:0.0000                3rd Qu.:0.0000      3rd Qu.:0.00000       
    ##  Max.   :1.0000                Max.   :1.0000      Max.   :1.00000       
    ##  data_channel_is_tech data_channel_is_world   kw_min_min       kw_max_min   
    ##  Min.   :0.0000       Min.   :0.0000        Min.   : -1.00   Min.   :    0  
    ##  1st Qu.:0.0000       1st Qu.:0.0000        1st Qu.: -1.00   1st Qu.:  484  
    ##  Median :0.0000       Median :0.0000        Median : -1.00   Median :  694  
    ##  Mean   :0.1457       Mean   :0.2057        Mean   : 29.59   Mean   : 1114  
    ##  3rd Qu.:0.0000       3rd Qu.:0.0000        3rd Qu.:  4.00   3rd Qu.: 1100  
    ##  Max.   :1.0000       Max.   :1.0000        Max.   :217.00   Max.   :81200  
    ##    kw_avg_min        kw_min_max       kw_max_max       kw_avg_max       kw_min_avg  
    ##  Min.   :   -1.0   Min.   :     0   Min.   : 37400   Min.   :  7328   Min.   :   0  
    ##  1st Qu.:  158.9   1st Qu.:     0   1st Qu.:843300   1st Qu.:168580   1st Qu.:   0  
    ##  Median :  243.8   Median :  1800   Median :843300   Median :231000   Median :1171  
    ##  Mean   :  320.3   Mean   : 12406   Mean   :748983   Mean   :242959   Mean   :1219  
    ##  3rd Qu.:  376.3   3rd Qu.:  8900   3rd Qu.:843300   3rd Qu.:307667   3rd Qu.:2169  
    ##  Max.   :27123.0   Max.   :690400   Max.   :843300   Max.   :762840   Max.   :3585  
    ##    kw_max_avg       kw_avg_avg      self_reference_min_shares self_reference_max_shares
    ##  Min.   :  2536   Min.   :  743.5   Min.   :     0            Min.   :     0           
    ##  1st Qu.:  3623   1st Qu.: 2497.0   1st Qu.:   674            1st Qu.:  1000           
    ##  Median :  4816   Median : 3040.8   Median :  1300            Median :  2700           
    ##  Mean   :  6060   Mean   : 3306.3   Mean   :  4199            Mean   : 10352           
    ##  3rd Qu.:  6864   3rd Qu.: 3855.7   3rd Qu.:  2700            3rd Qu.:  7600           
    ##  Max.   :120100   Max.   :15336.1   Max.   :843300            Max.   :843300           
    ##  self_reference_avg_sharess   is_weekend     LDA_00            LDA_01       
    ##  Min.   :     0.0           Min.   :1    Min.   :0.01824   Min.   :0.01820  
    ##  1st Qu.:   984.5           1st Qu.:1    1st Qu.:0.02500   1st Qu.:0.02500  
    ##  Median :  2150.8           Median :1    Median :0.03333   Median :0.03334  
    ##  Mean   :  6125.8           Mean   :1    Mean   :0.16610   Mean   :0.15677  
    ##  3rd Qu.:  5000.0           3rd Qu.:1    3rd Qu.:0.19850   3rd Qu.:0.17435  
    ##  Max.   :843300.0           Max.   :1    Max.   :0.92000   Max.   :0.92595  
    ##      LDA_02            LDA_03            LDA_04        global_subjectivity
    ##  Min.   :0.01818   Min.   :0.01818   Min.   :0.01945   Min.   :0.0000     
    ##  1st Qu.:0.02500   1st Qu.:0.02523   1st Qu.:0.02530   1st Qu.:0.3974     
    ##  Median :0.03334   Median :0.05000   Median :0.04000   Median :0.4604     
    ##  Mean   :0.19628   Mean   :0.26628   Mean   :0.21456   Mean   :0.4490     
    ##  3rd Qu.:0.27968   3rd Qu.:0.50891   3rd Qu.:0.32739   3rd Qu.:0.5148     
    ##  Max.   :0.92000   Max.   :0.91997   Max.   :0.92644   Max.   :0.9125     
    ##  global_sentiment_polarity global_rate_positive_words global_rate_negative_words
    ##  Min.   :-0.37393          Min.   :0.00000            Min.   :0.00000           
    ##  1st Qu.: 0.05948          1st Qu.:0.02817            1st Qu.:0.01004           
    ##  Median : 0.12061          Median :0.03951            Median :0.01615           
    ##  Mean   : 0.12294          Mean   :0.04099            Mean   :0.01706           
    ##  3rd Qu.: 0.18531          3rd Qu.:0.05283            3rd Qu.:0.02238           
    ##  Max.   : 0.63175          Max.   :0.13542            Max.   :0.10112           
    ##  rate_positive_words rate_negative_words avg_positive_polarity min_positive_polarity
    ##  Min.   :0.0000      Min.   :0.0000      Min.   :0.0000        Min.   :0.0000       
    ##  1st Qu.:0.6000      1st Qu.:0.1875      1st Qu.:0.3098        1st Qu.:0.0500       
    ##  Median :0.7059      Median :0.2857      Median :0.3667        Median :0.1000       
    ##  Mean   :0.6788      Mean   :0.2925      Mean   :0.3636        Mean   :0.0988       
    ##  3rd Qu.:0.8000      3rd Qu.:0.3846      3rd Qu.:0.4250        3rd Qu.:0.1000       
    ##  Max.   :1.0000      Max.   :1.0000      Max.   :1.0000        Max.   :1.0000       
    ##  max_positive_polarity avg_negative_polarity min_negative_polarity max_negative_polarity
    ##  Min.   :0.000         Min.   :-1.0000       Min.   :-1.0000       Min.   :-1.0000      
    ##  1st Qu.:0.600         1st Qu.:-0.3341       1st Qu.:-0.7071       1st Qu.:-0.1250      
    ##  Median :0.800         Median :-0.2615       Median :-0.5000       Median :-0.1000      
    ##  Mean   :0.777         Mean   :-0.2691       Mean   :-0.5405       Mean   :-0.1103      
    ##  3rd Qu.:1.000         3rd Qu.:-0.1984       3rd Qu.:-0.3333       3rd Qu.:-0.0500      
    ##  Max.   :1.000         Max.   : 0.0000       Max.   : 0.0000       Max.   : 0.0000      
    ##  title_subjectivity title_sentiment_polarity abs_title_subjectivity
    ##  Min.   :0.0000     Min.   :-1.0000          Min.   :0.0000        
    ##  1st Qu.:0.0000     1st Qu.: 0.0000          1st Qu.:0.1333        
    ##  Median :0.2667     Median : 0.0000          Median :0.4000        
    ##  Mean   :0.3163     Mean   : 0.0888          Mean   :0.3218        
    ##  3rd Qu.:0.5000     3rd Qu.: 0.2500          3rd Qu.:0.5000        
    ##  Max.   :1.0000     Max.   : 1.0000          Max.   :0.5000        
    ##  abs_title_sentiment_polarity     shares     
    ##  Min.   :0.0000               Min.   :  171  
    ##  1st Qu.:0.0000               1st Qu.: 1300  
    ##  Median :0.1000               Median : 1900  
    ##  Mean   :0.1857               Mean   : 3876  
    ##  3rd Qu.:0.3000               3rd Qu.: 3700  
    ##  Max.   :1.0000               Max.   :82000

``` r
correlation1 <- cor(train_data[,c(1:10,52)])
corrplot(correlation1,type='upper',tl.pos = 'lt')
corrplot(correlation1,type='lower',method = 'number',add = T,diag = F,tl.pos = 'n')
```

![](weekday_is_sunday_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

``` r
correlation2 <- cor(train_data[,c(11:20,52)])
corrplot(correlation2,type='upper',tl.pos = 'lt')
corrplot(correlation2,type='lower',method = 'number',add = T,diag = F,tl.pos = 'n')
```

![](weekday_is_sunday_files/figure-gfm/unnamed-chunk-6-2.png)<!-- -->

``` r
correlation3 <- cor(train_data[,c(21:30,52)])
```

    ## Warning in cor(train_data[, c(21:30, 52)]): the standard deviation is zero

``` r
corrplot(correlation3,type='upper',tl.pos = 'lt')
corrplot(correlation3,type='lower',method = 'number',add = T,diag = F,tl.pos = 'n')
```

![](weekday_is_sunday_files/figure-gfm/unnamed-chunk-6-3.png)<!-- -->

``` r
correlation4 <- cor(train_data[,c(31:40,52)])
corrplot(correlation4,type='upper',tl.pos = 'lt')
corrplot(correlation4,type='lower',method = 'number',add = T,diag = F,tl.pos = 'n')
```

![](weekday_is_sunday_files/figure-gfm/unnamed-chunk-6-4.png)<!-- -->

``` r
correlation5 <- cor(train_data[,c(41:51,52)])
corrplot(correlation5,type='upper',tl.pos = 'lt')
corrplot(correlation5,type='lower',method = 'number',add = T,diag = F,tl.pos = 'n')
```

![](weekday_is_sunday_files/figure-gfm/unnamed-chunk-6-5.png)<!-- -->

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
    ## -11697  -2597  -1314    223  68917 
    ## 
    ## Coefficients: (1 not defined because of singularities)
    ##                                 Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                    2.370e+03  1.450e+03   1.635 0.102265    
    ## n_tokens_title                 1.355e+02  7.323e+01   1.851 0.064329 .  
    ## n_tokens_content              -7.959e-02  4.525e-01  -0.176 0.860391    
    ## n_unique_tokens                8.015e+03  4.524e+03   1.772 0.076635 .  
    ## n_non_stop_words              -2.634e+03  4.040e+03  -0.652 0.514484    
    ## n_non_stop_unique_tokens      -4.766e+03  3.934e+03  -1.212 0.225831    
    ## num_hrefs                      5.539e+01  1.463e+01   3.785 0.000158 ***
    ## num_self_hrefs                -1.170e+02  3.695e+01  -3.168 0.001562 ** 
    ## num_imgs                       2.054e+01  2.032e+01   1.011 0.312134    
    ## num_videos                     6.282e+01  4.597e+01   1.367 0.171934    
    ## average_token_length          -8.054e+02  5.708e+02  -1.411 0.158447    
    ## num_keywords                   2.014e+02  8.175e+01   2.464 0.013837 *  
    ## data_channel_is_lifestyle     -1.139e+03  6.529e+02  -1.745 0.081104 .  
    ## data_channel_is_entertainment -6.948e+02  5.155e+02  -1.348 0.177832    
    ## data_channel_is_bus           -6.520e-01  6.536e+02  -0.001 0.999204    
    ## data_channel_is_socmed         7.439e+02  7.983e+02   0.932 0.351520    
    ## data_channel_is_tech          -1.971e+02  5.791e+02  -0.340 0.733611    
    ## data_channel_is_world         -1.301e+03  5.884e+02  -2.211 0.027156 *  
    ## self_reference_min_shares     -6.343e-03  2.856e-02  -0.222 0.824248    
    ## self_reference_max_shares     -5.209e-03  8.454e-03  -0.616 0.537849    
    ## self_reference_avg_sharess     2.094e-02  3.525e-02   0.594 0.552591    
    ## global_subjectivity            2.136e+03  2.041e+03   1.046 0.295495    
    ## global_sentiment_polarity      6.999e+03  3.941e+03   1.776 0.075869 .  
    ## global_rate_positive_words    -5.768e+04  1.645e+04  -3.506 0.000466 ***
    ## global_rate_negative_words     5.043e+04  3.421e+04   1.474 0.140621    
    ## rate_positive_words            3.600e+03  2.835e+03   1.270 0.204261    
    ## rate_negative_words                   NA         NA      NA       NA    
    ## avg_positive_polarity          6.550e+00  3.364e+03   0.002 0.998447    
    ## min_positive_polarity         -3.928e+03  2.788e+03  -1.409 0.158990    
    ## max_positive_polarity          9.705e+01  1.138e+03   0.085 0.932031    
    ## avg_negative_polarity         -6.479e+02  3.041e+03  -0.213 0.831338    
    ## min_negative_polarity         -1.013e+03  1.118e+03  -0.906 0.364836    
    ## max_negative_polarity         -4.408e+02  2.551e+03  -0.173 0.862820    
    ## title_subjectivity             7.519e+02  7.082e+02   1.062 0.288559    
    ## title_sentiment_polarity       7.536e+01  6.102e+02   0.123 0.901731    
    ## abs_title_subjectivity         1.079e+03  9.112e+02   1.185 0.236335    
    ## abs_title_sentiment_polarity  -3.231e+02  9.775e+02  -0.331 0.741012    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 6439 on 1879 degrees of freedom
    ## Multiple R-squared:  0.05251,    Adjusted R-squared:  0.03486 
    ## F-statistic: 2.975 on 35 and 1879 DF,  p-value: 1.673e-08

Then we will fit some candidate models to select.

``` r
# Candidate models
fit1 <- train(shares ~ num_hrefs, train_data,
             method='lm',
             trControl = trainControl(method = 'cv',number=10))
fit1$results$RMSE
```

    ## [1] 6438.438

``` r
fit2 <- train(shares ~ num_hrefs+average_token_length, train_data,
             method='lm',
             trControl = trainControl(method = 'cv',number=10))
fit2$results$RMSE
```

    ## [1] 6291.765

``` r
fit3 <- train(shares ~ num_hrefs+average_token_length
              +data_channel_is_lifestyle, train_data,
             method='lm',
             trControl = trainControl(method = 'cv',number=10))
fit3$results$RMSE
```

    ## [1] 6277.516

``` r
fit4 <- train(shares ~ num_hrefs+average_token_length
              +data_channel_is_lifestyle+data_channel_is_entertainment, train_data,
             method='lm',
             trControl = trainControl(method = 'cv',number=10))
fit4$results$RMSE
```

    ## [1] 6329.843

``` r
fit5 <- train(shares ~ num_hrefs+average_token_length
              +data_channel_is_lifestyle+data_channel_is_entertainment
              +data_channel_is_bus, train_data,
             method='lm',
             trControl = trainControl(method = 'cv',number=10))
fit5$results$RMSE
```

    ## [1] 6489.355

``` r
fit6 <- train(shares ~ num_hrefs+average_token_length
              +data_channel_is_lifestyle+data_channel_is_entertainment
              +data_channel_is_bus+data_channel_is_socmed, train_data,
             method='lm',
             trControl = trainControl(method = 'cv',number=10))
fit6$results$RMSE
```

    ## [1] 6390.849

``` r
fit7<- train(shares ~ num_hrefs+average_token_length
              +data_channel_is_lifestyle+data_channel_is_entertainment
              +data_channel_is_bus+data_channel_is_socmed+data_channel_is_tech, train_data,
             method='lm',
             trControl = trainControl(method = 'cv',number=10))
fit7$results$RMSE
```

    ## [1] 6424.317

``` r
fit8 <- train(shares ~ num_hrefs+average_token_length
              +data_channel_is_lifestyle+data_channel_is_entertainment
              +data_channel_is_bus+data_channel_is_socmed+data_channel_is_tech+
              data_channel_is_world,
              train_data,method='lm',
             trControl = trainControl(method = 'cv',number=10))
fit8$results$RMSE
```

    ## [1] 6314.397

``` r
fit9 <- train(shares ~ num_hrefs+average_token_length
              +data_channel_is_lifestyle+data_channel_is_entertainment
              +data_channel_is_bus+data_channel_is_socmed+data_channel_is_tech+
              data_channel_is_world+self_reference_min_shares,
              train_data,method='lm',
             trControl = trainControl(method = 'cv',number=10))
fit9$results$RMSE
```

    ## [1] 6334.174

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

    ## [1] 6470.624

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
    ## [1,]    6490.786       6326.148     6470.624

In this case,we can see the linear model generates the smallest RMSE.The
linear model tend to have a better prediction than the tree based method
and boosted method.
