TREND
================

# Deterministic trend

DEF. Y_t = μ_t + Z_t is a process with determinisitc trend.  
μ_t is the deterministic func. of t i.e. deterministic trend, and Z_t is
a stationary process with E(Z_t)=0

1.  a deterministic trend measures the change in the mean function,
    i.e. calculate mean at eacn t  
2.  AR(1) with trend: Y_t - μ = φ\*(Y_t-1 - μ) + e_t  
3.  MA(1) with trend: Y_t - μ = e_t + θ\*e_t-1

### Estimate the trend μ (μ_hat) and var(μ_hat)

1.  Estimation of μ can be biased since autocorr., we shouldn’t use
    simple average to estimate μ, rather, we should use MLE approach
    provided by R to avoid this issue  
2.  estimation of var(μ_hat) can also be biased when autocorr. is strong

# Fit a linear model to a random walk

Means that it’s actually a rw but you write a linear expression to fit.

### Spurious regression

Random walk has E(Y_t)=0 and increasing variance, lm will show
significant trend when there’s not, failing to distinguish rw from
linear trend

In business, “how to distinguish linear trend from rw” is more important
than “how to fit a linear trend”

##### How to solve?

1.  Regression diagnostics: residual diagnostics
    1)  residual plot: if fitting a trend nicely, then residuals should
        look like white noise  
    2)  ACF: a linear trend shouldn’t have significant autocorr., while
        rw does  
    3)  QQ plot: check normality of the residuals. Note that being
        normally distributed is a stronger condition than just being
        independent noise, therefore sometimes we may see independent
        residuals in the residual plot and ACF but no straight line in
        the QQ plot. In this case we still believe the underlying
        pattern of the time series has a deterministic trend, but just
        the model does not fit that well.
2.  unit-root tests (ADF test)
    1)  rw: p-value \>= 0.05  
    2)  linear trend: p-value \< 0.05  
        One may wonder that a stochastic process with linear trend is
        not stationary since the mean is changing over time, yet ADF
        takes care of the mean automatically and only considers the
        residuals of the trend

# Seasonal trend

1.  Seasonal ARIMA only captures seasonal autocorr., if we have domain
    knowledge that the seasonality is deterministic i.e. there’s a
    deterministic seasonal trend, we may consider seasonality as another
    type of the trend, then consider fitting the seasonality with a
    deterministic seasonal trend as oppose to seasonal ARIMA.  
2.  Without domain knowledge, we can check the residual plot, QQ plot,
    and the ACF to evaluate seasonal dependency after removing the
    seasonal trend.

### Cosine trend (cycle)

Take seasonal trend within a year as example, the general idea is to use
cosine expression to capture the non-linear seasonal trend s.t. we don’t
have to specify 12 dummy variables(μi, i=1 to 12) and will only have 2
(sin and cos), saving a lot cost for variables.

# Sequential modeling

Fit multiple models sequentially rather than simultaneously.  
Examples:  
1. Tensor factorization + LSTM  
2. Image recognition + logistic regression  
3. Embedding + XGBoost  
4. Text sentiment analysis + ARIMA

# Codes

### Deterministic linear trend vs. Random Walk

``` r
set.seed(8) 
N=150
X=1:N
Y=-1-0.1*X+rnorm(N) # deterministic linear trend
Y2=cumsum(rnorm(N)) # random walk
par(mfrow=c(1,2)) 
plot(Y,type="o", xlab = "Time")
plot(Y2,type="o", xlab = "Time")
```

![](7_trend_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->

``` r
# split the data into a training set and a testing set: 100 training data, 50 testing data
# forecast horizon quite long
Ytrain=Y[1:100]
Ytest=Y[101:N]

Y2train=Y2[1:100]
Y2test=Y2[101:N]

Xtrain=X[1:100]
Xtest=X[101:N]
```

### Model specification

Check the consequence of model misspecification in terms of both
inference and forecasting

##### Correct specification: fit a linear trend with a linear model

``` r
# inference 
Xlm=Xtrain
lm1=lm(Ytrain~Xlm)
summary(lm1) # correct inference results
```

    ## 
    ## Call:
    ## lm(formula = Ytrain ~ Xlm)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -2.95659 -0.62983  0.02472  0.70611  2.49608 
    ## 
    ## Coefficients:
    ##              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept) -1.046031   0.218386   -4.79 5.93e-06 ***
    ## Xlm         -0.100936   0.003754  -26.89  < 2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 1.084 on 98 degrees of freedom
    ## Multiple R-squared:  0.8806, Adjusted R-squared:  0.8794 
    ## F-statistic: 722.8 on 1 and 98 DF,  p-value: < 2.2e-16

``` r
# forecast
library(forecast)
```

    ## Warning: package 'forecast' was built under R version 4.3.2

    ## Registered S3 method overwritten by 'quantmod':
    ##   method            from
    ##   as.zoo.data.frame zoo

``` r
X_test=data.frame(Xlm=Xtest) # a column named Xlm to store Xtest 
lm1_forecast=forecast(lm1, newdata = X_test)
mu_forecast=lm1_forecast$mean
mu_forecast=ts(mu_forecast,start = 101,end = N)
ts.plot(Y)
lines(c(lm1$fitted.values),col="orange")
lines(c(rep(NA,length(Ytrain)),mu_forecast),col="red",lwd="3") # good forecasts
```

![](7_trend_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

##### Correct specification: fit a random walk with a random walk

``` r
rw2=Arima(Y2train, order=c(0,1,0),method="ML") # recall that Y2 is rw 
rw2 # no parameter estimation, so no inference, (but smaller AIC/BIC than the previous model rw1)
```

    ## Series: Y2train 
    ## ARIMA(0,1,0) 
    ## 
    ## sigma^2 = 0.7605:  log likelihood = -126.92
    ## AIC=255.85   AICc=255.89   BIC=258.44

``` r
# forecast
rw2_pred <- forecast(rw2,h=N-100)
autoplot(rw2_pred) +
  autolayer(ts(Y2test,start=101,end=N), series="Data") +
  autolayer(rw2_pred$mean, series="Forecasts") 
```

![](7_trend_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

``` r
# we can't forecast a random walk, but at least the 95% prediction interval covers roughly 95% of the future values
```

##### Misspecification: fit a random walk with a linear model

``` r
# inference 
lm2=lm(Y2train~Xlm)
summary(lm2) # false positive results: should be non-significant
```

    ## 
    ## Call:
    ## lm(formula = Y2train ~ Xlm)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -3.4794 -1.1655 -0.1537  0.9318  5.3822 
    ## 
    ## Coefficients:
    ##             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept) 3.039597   0.388374   7.826 5.94e-12 ***
    ## Xlm         0.017716   0.006677   2.653   0.0093 ** 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 1.927 on 98 degrees of freedom
    ## Multiple R-squared:  0.06702,    Adjusted R-squared:  0.0575 
    ## F-statistic:  7.04 on 1 and 98 DF,  p-value: 0.009299

``` r
# forecast
lm2_forecast=forecast(lm2,newdata = X_test)
mu_forecast2=lm2_forecast$mean
mu_forecast2=ts(mu_forecast2,start = 101,end = N)
ts.plot(Y2)
lines(c(lm2$fitted.values),col="orange")
lines(c(rep(NA,length(Ytrain)),mu_forecast2),col="red",lwd="3") # poor forecasts
```

![](7_trend_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

##### Misspecification: fit a linear trend with a random walk

``` r
rw1=Arima(Ytrain, order=c(0,1,0), method="ML") # recall that Y is the process with deterministic trend 
rw1 # no parameter estimation, so no inference
```

    ## Series: Ytrain 
    ## ARIMA(0,1,0) 
    ## 
    ## sigma^2 = 2.685:  log likelihood = -189.37
    ## AIC=380.73   AICc=380.77   BIC=383.33

``` r
# forecast
rw1_pred <- forecast(rw1, h=N-100)
autoplot(rw1_pred) +
  autolayer(ts(Ytest,start=101,end=N), series="Data") +
  autolayer(rw1_pred$mean, series="Forecasts") # systematic bias in forecasting
```

![](7_trend_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

### Differentiation between a linear trend and a random walk

``` r
# re-run the regression with all data (can be skipped and use the previous lm1 and lm2)
LM1=lm(Y~X) # Y1 is linear trend 
LM2=lm(Y2~X) # Y2 is rw

# check the ADF test
library(tseries)
```

    ## Warning: package 'tseries' was built under R version 4.3.2

``` r
adf.test(LM1$residuals) # p-value significant 
```

    ## Warning in adf.test(LM1$residuals): p-value smaller than printed p-value

    ## 
    ##  Augmented Dickey-Fuller Test
    ## 
    ## data:  LM1$residuals
    ## Dickey-Fuller = -5.4234, Lag order = 5, p-value = 0.01
    ## alternative hypothesis: stationary

``` r
adf.test(LM2$residuals) # p-value not significant 
```

    ## 
    ##  Augmented Dickey-Fuller Test
    ## 
    ## data:  LM2$residuals
    ## Dickey-Fuller = -1.3459, Lag order = 5, p-value = 0.8489
    ## alternative hypothesis: stationary

``` r
# check the regression residuals
par(mfrow=c(1,2)) 
plot(LM1$residuals,type="o",xlab = "Time",ylab = "Linear Model Residuals")
abline(h=0,col="red") # independent model residuals
plot(LM2$residuals,type="o",xlab = "Time",ylab = "Linear Model Residuals")
abline(h=0,col="red") # autocorrelated model residuals
```

![](7_trend_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

``` r
# QQ plot (for normality of the residuals)
par(mfrow=c(1,2)) 
qqnorm(LM1$residuals) # circles align with the solid line
qqline(LM1$residuals, col = "steelblue", lwd = 2)
qqnorm(LM2$residuals) # circles do not align with the solid line
qqline(LM2$residuals, col = "steelblue", lwd = 2)
```

![](7_trend_files/figure-gfm/unnamed-chunk-6-2.png)<!-- -->

``` r
# we can also check the ACF of the residuals
Linear_Trend_res=LM1$residuals
Random_Walk_res=LM2$residuals
par(mfrow=c(1,2)) 
Acf(Linear_Trend_res) # independent model residuals shown on ACF: no significant autocorr. 
Acf(Random_Walk_res)  # autocorrelated model residuals shown on ACF: significant autocorr. tails off
```

![](7_trend_files/figure-gfm/unnamed-chunk-6-3.png)<!-- -->

### Seasonal ARIMA vs. A seasonal trend

We consider a set of monthly temperature data, fitting it with both a
seasonal trend and a seasonal ARIMA.  
Then we run model diagnostics, and forecast 12 months ahead.

``` r
# Load the data: monthly Dublin temperature 
library(TSA)
```

    ## Warning: package 'TSA' was built under R version 4.3.2

    ## Registered S3 methods overwritten by 'TSA':
    ##   method       from    
    ##   fitted.Arima forecast
    ##   plot.Arima   forecast

    ## 
    ## Attaching package: 'TSA'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     acf, arima

    ## The following object is masked from 'package:utils':
    ## 
    ##     tar

``` r
data(tempdub)
month. = season(tempdub)
par(mfrow=c(1,1)) 
ts.plot(tempdub)
```

![](7_trend_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

``` r
length(tempdub) # 144 months
```

    ## [1] 144

``` r
# Training testing split 
temp_train=tempdub[1:132]
temp_test=tempdub[133:144] # testing set: predict 12 months ahead 
```

##### Seasonal Trend

First try fitting the data with a linear model using 12 dummy variables
with each dummy variable indicates a month

``` r
# Fit the linear model 
month_ = month.[1:132]
seasonal_lm = lm(temp_train ~ 0 + month_)
summary(seasonal_lm)
```

    ## 
    ## Call:
    ## lm(formula = temp_train ~ 0 + month_)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -8.9909 -2.2568  0.0136  1.7341  9.1091 
    ## 
    ## Coefficients:
    ##                 Estimate Std. Error t value Pr(>|t|)    
    ## month_January     16.264      1.018   15.98   <2e-16 ***
    ## month_February    20.745      1.018   20.38   <2e-16 ***
    ## month_March       33.191      1.018   32.61   <2e-16 ***
    ## month_April       47.000      1.018   46.18   <2e-16 ***
    ## month_May         57.755      1.018   56.75   <2e-16 ***
    ## month_June        67.409      1.018   66.23   <2e-16 ***
    ## month_July        71.691      1.018   70.44   <2e-16 ***
    ## month_August      69.173      1.018   67.97   <2e-16 ***
    ## month_September   61.364      1.018   60.30   <2e-16 ***
    ## month_October     50.836      1.018   49.95   <2e-16 ***
    ## month_November    36.291      1.018   35.66   <2e-16 ***
    ## month_December    23.409      1.018   23.00   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 3.375 on 120 degrees of freedom
    ## Multiple R-squared:  0.9959, Adjusted R-squared:  0.9955 
    ## F-statistic:  2416 on 12 and 120 DF,  p-value: < 2.2e-16

``` r
# Check the residual plot
plot(seasonal_lm$residuals, type="o", xlab = "Time", ylab = "Dummy Variable Model Residuals") # looks OK
```

![](7_trend_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

``` r
# Check the ADF test result
adf.test(seasonal_lm$residuals) # stationary
```

    ## Warning in adf.test(seasonal_lm$residuals): p-value smaller than printed
    ## p-value

    ## 
    ##  Augmented Dickey-Fuller Test
    ## 
    ## data:  seasonal_lm$residuals
    ## Dickey-Fuller = -5.0316, Lag order = 5, p-value = 0.01
    ## alternative hypothesis: stationary

``` r
# Check the QQ plot
qqnorm(seasonal_lm$residuals) # circles roughly align with the solid line
qqline(seasonal_lm$residuals, col = "steelblue", lwd = 2)
```

![](7_trend_files/figure-gfm/unnamed-chunk-8-2.png)<!-- -->

``` r
# Check the ACF plot
Acf(seasonal_lm$residuals) # independent model residuals shown on ACF 
```

![](7_trend_files/figure-gfm/unnamed-chunk-8-3.png)<!-- -->

``` r
# All of the above diagnostics show independent model residuals, 
# indicating that the original data may not contain a random walk component 

# Forecast with the above seasonal trend
mu_forecast = as.vector(seasonal_lm$coefficients) # for dummy variables, we can directly use the coefficients as the predicted values for the corresponding months
mu_forecast = ts(mu_forecast, start = 133, end = 144)
ts.plot(tempdub[1:144])
lines(c(seasonal_lm$fitted.values), col="orange")
lines(c(rep(NA, length(temp_train)), mu_forecast), col="red",lwd="3") 
```

![](7_trend_files/figure-gfm/unnamed-chunk-8-4.png)<!-- -->

``` r
# good forecasts, align with the true value almost perfectly 

# Check the forecasting accuracy with the root mean square error
library(Metrics)
```

    ## Warning: package 'Metrics' was built under R version 4.3.2

    ## 
    ## Attaching package: 'Metrics'

    ## The following object is masked from 'package:forecast':
    ## 
    ##     accuracy

``` r
rmse(tempdub[133:144],mu_forecast) # 3.997
```

    ## [1] 3.997075

##### Seasonal ARIMA

Next the data are fit with auto.arima()

``` r
# Fit the model 
seasonal_arima = auto.arima(temp_train, seasonal = TRUE)
seasonal_arima 
```

    ## Series: temp_train 
    ## ARIMA(4,0,0) with non-zero mean 
    ## 
    ## Coefficients:
    ##          ar1    ar2      ar3      ar4     mean
    ##       0.8239  0.024  -0.0171  -0.4762  46.2429
    ## s.e.  0.0761  0.105   0.1050   0.0764   0.6591
    ## 
    ## sigma^2 = 24.17:  log likelihood = -397.48
    ## AIC=806.95   AICc=807.63   BIC=824.25

``` r
# Note that seasonal ARIMA only captures seasonal autocorr.  
# Here auto.arima() doesn't use any seasonal components, although "seasonal=TRUE" is clearly specified 

# Check the residual plot
plot(seasonal_arima$residuals, type="o", xlab = "Time", ylab = "Dummy Variable Model Residuals") # looks OK
```

![](7_trend_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

``` r
# Check the ADF test result
adf.test(seasonal_arima$residuals) # stationary
```

    ## Warning in adf.test(seasonal_arima$residuals): p-value smaller than printed
    ## p-value

    ## 
    ##  Augmented Dickey-Fuller Test
    ## 
    ## data:  seasonal_arima$residuals
    ## Dickey-Fuller = -6.8016, Lag order = 5, p-value = 0.01
    ## alternative hypothesis: stationary

``` r
# Check the QQ plot
qqnorm(seasonal_arima$residuals) # circles roughly align with the solid line
qqline(seasonal_arima$residuals, col = "steelblue", lwd = 2)
```

![](7_trend_files/figure-gfm/unnamed-chunk-9-2.png)<!-- -->

``` r
# Check the ACF plot
Acf(seasonal_arima$residuals) # seasonally autocorr. residuals shown on ACF though autocorr. is not particularly strong
```

![](7_trend_files/figure-gfm/unnamed-chunk-9-3.png)<!-- -->

``` r
# Compared with the seasonal trend, 
# the ARIMA model selected by auto.arima() doesn't take care of the seasonality perfectly

# Forecasting 
arima_pred <- forecast(seasonal_arima, h=144-132)
autoplot(arima_pred) +
  autolayer(ts(temp_test,start=133,end=144), series="Data") +
  autolayer(arima_pred$mean, series="Forecasts") 
```

![](7_trend_files/figure-gfm/unnamed-chunk-9-4.png)<!-- -->

``` r
# The result is not as perfect as that of the seasonal trend, 
# but at least the 95% prediction interval covers roughly 95% of the future values

# Check the forecasting rmse, worse than the seasonal trend
rmse(tempdub[133:144],arima_pred$mean) # 4.832
```

    ## [1] 4.831682

### Example of Overfitting

##### Overfitting

The following chunk demonstrates overfitting situation

``` r
# Generate a series of white noise
set.seed(66)
Y=rnorm(1000)

# Split the data into 950 points for training and 50 points for testing
Y_train=Y[1:950]
Y_test=Y[951:1000]

# Check stationarity: stationary 
adf.test(Y_train)
```

    ## Warning in adf.test(Y_train): p-value smaller than printed p-value

    ## 
    ##  Augmented Dickey-Fuller Test
    ## 
    ## data:  Y_train
    ## Dickey-Fuller = -10.365, Lag order = 9, p-value = 0.01
    ## alternative hypothesis: stationary

``` r
# Deliberately choose a random walk as the (misspecified) model, in order to check overfitting
rw2 = Arima(Y_train, order=c(0,1,0))

# Check the goodness of fit: Visually it fits perfectly
ts.plot(c(Y_train))
lines(c(rw2$fitted),col="red")
```

![](7_trend_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

``` r
# However, if we take a closer look, say, checking only the first 100 points, 
# we see that there's always a lag between the fitted value and the true value
ts.plot(c(Y_train)[1:100])
lines(c(rw2$fitted)[1:100],col="red")
```

![](7_trend_files/figure-gfm/unnamed-chunk-10-2.png)<!-- -->

``` r
# We use the fitted random walk to forecast the next 50 points
rw2_pred <- forecast(rw2,h=50)
# We can see it fits poorly
autoplot(rw2_pred) +
  autolayer(ts(Y_test,start=951,end=1000), series="Data") +
  autolayer(rw2_pred$mean, series="Forecasts") 
```

![](7_trend_files/figure-gfm/unnamed-chunk-10-3.png)<!-- -->

``` r
# Check the RMSE
library(Metrics)
rmse(Y_test,rw2_pred$mean) # 1.739
```

    ## [1] 1.73871

##### Correct way

The following chunk demonstrates the correct way to do it

``` r
# Check stationarity: significant, and hence ARMA(p,q) can be adopted
adf.test(Y_train)
```

    ## Warning in adf.test(Y_train): p-value smaller than printed p-value

    ## 
    ##  Augmented Dickey-Fuller Test
    ## 
    ## data:  Y_train
    ## Dickey-Fuller = -10.365, Lag order = 9, p-value = 0.01
    ## alternative hypothesis: stationary

``` r
# Choose proper p and q
Acf(Y_train)  # suggests q=0, although lag=5 is weakly significant (considered as false positive)
```

![](7_trend_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

``` r
Pacf(Y_train) # suggests p=0, although lag=5 is weakly significant (considered as false positive)
```

![](7_trend_files/figure-gfm/unnamed-chunk-11-2.png)<!-- -->

``` r
eacf(Y_train) # suggests (p,q)=(0,0)
```

    ## AR/MA
    ##   0 1 2 3 4 5 6 7 8 9 10 11 12 13
    ## 0 o o o o x o o o o o o  o  o  o 
    ## 1 x o o o x o o o o o o  o  o  o 
    ## 2 x x o o x o o o o o o  o  o  o 
    ## 3 o x o o x o o o o o o  o  o  o 
    ## 4 o x x x o o o o o o o  o  x  o 
    ## 5 x x x x x o o o o o o  o  o  o 
    ## 6 x x o x x x o o o o o  o  o  o 
    ## 7 x o o o x x x o o o o  o  o  o

``` r
# Therefore, we fit a white noise to it, which sounds surprisingly trivial, but is true in this case
rw1 = Arima(Y_train,order=c(0,0,0))

# Check the fitted line: although the goodness of fit is poorer than that of the random walk, but this is the true model
ts.plot(c(Y_train))
lines(c(rw1$fitted), col="red")
```

![](7_trend_files/figure-gfm/unnamed-chunk-11-3.png)<!-- -->

``` r
# We use the fitted white noise (which is basically just an estimated mean) to forecast the next 50 points
rw1_pred <- forecast(rw1, h=50)
autoplot(rw1_pred) +
  autolayer(ts(Y_test,start=951,end=1000), series="Data") +
  autolayer(rw1_pred$mean, series="Forecasts") 
```

![](7_trend_files/figure-gfm/unnamed-chunk-11-4.png)<!-- -->

``` r
# Check the RMSE, better than the random walk
rmse(Y_test,rw1_pred$mean) # 0.998
```

    ## [1] 0.9981791

### Supplementary Materials

##### Mean Estimation for MA(1)

``` r
y=arima.sim(model=list(order=c(0,0,1),ma=0.95,sd=1),n=1000)
mean(y)
```

    ## [1] -0.04750133

``` r
y=arima.sim(model=list(order=c(0,0,1),ma=-0.95,sd=1),n=1000)
mean(y)
```

    ## [1] -0.001551702

##### Linear Trend with White Noise

``` r
N=200
set.seed(88)
e=rnorm(N,0,5)
t=1:N
Y=ts(2 + t + e)
plot(Y,type='o')
```

![](7_trend_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

``` r
summary(lm(Y~t))
```

    ## 
    ## Call:
    ## lm(formula = Y ~ t)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -14.3572  -3.5390   0.1126   3.2775  12.6835 
    ## 
    ## Coefficients:
    ##             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept) 1.441159   0.707906   2.036   0.0431 *  
    ## t           1.006952   0.006108 164.865   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 4.987 on 198 degrees of freedom
    ## Multiple R-squared:  0.9928, Adjusted R-squared:  0.9927 
    ## F-statistic: 2.718e+04 on 1 and 198 DF,  p-value: < 2.2e-16

##### Linear Trend with Moving Average

``` r
set.seed(88)
e=rnorm(N,0,5)
X=ts(e+0.9*zlag(e))
ts.plot(X)
```

![](7_trend_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->

``` r
t=1:N
Y=ts(2 + t + X)
plot(Y,type='o')
```

![](7_trend_files/figure-gfm/unnamed-chunk-14-2.png)<!-- -->

``` r
summary(lm(Y~t))
```

    ## 
    ## Call:
    ## lm(formula = Y ~ t)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -16.9912  -4.9948   0.0689   4.8241  20.6452 
    ## 
    ## Coefficients:
    ##             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept) 0.940700   1.003340   0.938     0.35    
    ## t           1.013166   0.008635 117.332   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 6.998 on 197 degrees of freedom
    ##   (1 observation deleted due to missingness)
    ## Multiple R-squared:  0.9859, Adjusted R-squared:  0.9858 
    ## F-statistic: 1.377e+04 on 1 and 197 DF,  p-value: < 2.2e-16

##### Hybrid Trend

``` r
set.seed(88)
e=rnorm(N,0,10)
t=1:N
Y=ts(2 + t + 3*cos(t) + e)
ts.plot(Y)
```

![](7_trend_files/figure-gfm/unnamed-chunk-15-1.png)<!-- -->
