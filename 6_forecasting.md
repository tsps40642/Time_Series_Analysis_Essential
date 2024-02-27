FORECASTING
================

In this chapter, will discuss multiple approaches for forecasting

# LOCF & Average forecasting

### LOCF, last observation carried forward

DEF. Y_t+h_hat = Y_t, ∀h \> 0

It forecast all future values as the latest observation collected. h is
the forecast horizon, the length of the time into the future for wich
forecast are to be prepared

### Average forecasting

DEF. Y_t+h_hat = (Σ(i=0 to T-1) Y_t-i)/T, ∀h \> 0

### LOCF and Average forecasting can bith be written as

DEF. Y_t+h_hat = Σ(i=0 to T-1) αi\*Y_t-i

1.  LOCF: α0 = 1 and αi = 0 for any i  
2.  Average: αi = 1/T for any i  
3.  LOCF and Average are two extreme cases, we allowed more flexible
    weights αi’s

# Simple exponential smoothing

DEF. Y_t+1_hat = α*Y_t + α(1-α)*Y_t-1 + α(1-α)<sup>2</sup>\*Y_t-2 + …

1.  0\<=α\<=1 is a smoothing parameter  
2.  Weights decay exponentially, very popular in practice  
3.  Other additive smoothing models are special cases of ARIMA

# AR forecasting

## AR(1) forecasting

Given Y_t = φ\*Y_t-1 + e_t, φ is some constant

1.  We can approximate Y_t+1_hat as Y_t+1_hat = φ_hat\*Y_t + e_t+1_hat.
    We have φ_hat, and e_t+1_hat needs to be estimated  
2.  e_t+1 is the white noise at t+1. By MLE, e_t+1_hat = E(e_t+1) = 0  
3.  Note that since lacking prior info., we assume e_t ~ N(0,
    σ<sup>2</sup>), if we have domain knowledge, we might use other
    distribution in e_t  
4.  Based on above, it becomes Y_t+1_hat = φ_hat\*Y_t  
5.  Y_t+2 = φ\*Y_t+1 + e_t+2, where we already know to use E(e_t+2) to
    estimate e_t+2  
6.  Thus we use φ*Y_t+1_hat to estimate: Y_t+2_hat = φ*Y_t+1_hat, the
    same for Y_t+3_hat,…  
7.  And if we are provided with true value by any chance, use the true
    rather than estimated

## AR(2) forecasting

(see course slide)

## AR will have converging pattern

Since abs(φ)\<1, abs(φ_hat)\<1, that means Y_t+h_hat =
φ_hat<sup>h</sup>\*Y_t goes to 0 as h increase

## Forecasting with non-zero mean

Y_t+1_hat = μ_hat + φ_hat\*(Y_t - μ_hat)

# MA forecasting

## MA(1) forecasting

1.  Y_t+1 = e_t+1 + θ\*e_t. We have θ_hat, and recall that e_t+1 is
    white noise so replace with E(e_t+1)=0  
2.  Then it becomes Y_t+1 = θ_hat\*e_t, where e_t_hat = Y_t - Y_t_hat  
3.  Then it becomes Y_t+1_hat = θ_hat\*(Y_t - Y_t_hat), where Y_1_hat,
    …, Y_t_hat are calculated by arima  
4.  Note that for Y_1, …, Y_t, we already have their true values,
    getting Y_1_hat, …, Y_t_hat helps us get residuals for forecasting
    and model diagnosis  
5.  Y_t+2_hat = θ_hat\*(Y_t+1 - Y_t+1_hat) and we use Y_t+1_hat to
    estimate Y_t+1 so Y_t+2_hat=0  
6.  consider 5., and can also explained by the form of MA: Y_t+2 =
    θ\*e_t+1 + e_t+2 and both are not observed and are approximated by
    expected values=0

## Forecasting with non-zero mean

Y_t+1_hat-μ_hat = θ_hat*\[(Y_t-μ_hat) - (Y_t_hat-μ_hat)\], hance 1.
Y_t+1_hat = μ_hat + θ_hat*(Y_t-Y_t_hat)  
2. Y_t+h_hat = μ_hat, ∀h \> 2  
3. Based on 2. that is, MA(1) can only forecast informatively at most
one point ahead

## MA(2) forecasting

1.  Y_t+1_hat = μ_hat + θ1_hat*(Y_t-Y_t_hat) +
    θ2_hat*(Y_t-1-Y_t-1_hat)  
2.  Y_t+2_hat = μ_hat + θ3_hat*(Y_t-Y_t_hat) + θ4_hat*(Y_t+1-Y_t+1_hat)
    and since Y_t+1 is estimated by Y_t+1_hat, it becomes Y_t+2_hat =
    μ_hat + θ3_hat\*(Y_t-Y_t_hat)  
3.  Y_t+3_hat = μ_hat the same reason as above  
4.  Extend to MA(q), it can’t forecast more than q points ahead \>\>\>
    short memory  
5.  So the plot of forecast MA(q) will be 0 or μ (depends on having
    non-zero mean or not) after q points

## ARMA(p, q) forecasting

(see course slides p.48)

# Predictive interval (PI)

1.  95% PI: Y_t_hat +- 1.96\*σ_t_hat  
2.  80% PI: Y_t_hat +- 1.28\*σ_t_hat, σ_t_hat is the estimated sd

# ARMAX

ARMA or ARIMA model with X, the vector of covariates at time t
(time-invariant or time-dependent) with its coeff. β

### Potential problem 1 in ARMAX

Considering rewrite the model with backshift operator: Φ(B)*Y_t =
β*X_t + Θ(B)\*e_t, then β/Φ(B) is the actual effect of X_t on Y_t, not β
as in the linear regression

##### Solution

1.  Fit (Y_t~β\*X_t) first s.t. β has regular interpretation  
2.  Then fit the residual (Y_t~β_hat\*X_t) with ARIMA models (more
    details about trend will be in TREND)

### Potential problem 2 in ARMAX

We don’t necessarily have feature T every time, and if X_t depends on t,
then we don’t know X_T+1_hat

##### Solution

Use lagged X_T

# Validation on a rolling basis

1.  Size of forecasting is defined by domain knowledge  
2.  Training set size is not fixed, while testing set size is fixed  
3.  This transforms a time series problem to a supervised learning
    problem

##### Example: we have Y_1, …, Y_100 and want to predict Y_101_hat, …, Y_105_hat

1.  Train the model  
    train (Y_1, …, Y_10), test (Y_11_hat, …Y_15_hat)  
    train (Y_1, …, Y_15), test (Y_16_hat, …Y_20_hat)  
    …  
    train (Y_1, …, Y_95), test (Y_96_hat, …Y_100_hat)  
2.  Select the model with the best performance on Y_11_hat, …,
    Y_100_hat  
3.  Retrain with Y_1, …Y_100 and forecast Y_101_hat, …Y_105_hat

##### Example: we have Y_1, …, Y_100 and X_1, …X_100, and we know Y_t is associated with X_t-12

1.  Train the model  
    train (Y_13, X_1), …, (Y_30, X_18), test (Y_31_hat, X_19), …,
    (Y_35_hat, X_23)  
    train (Y_13, X_1), …, (Y_35, X_23), test (Y_36_hat, X_24), …,
    (Y_40_hat, X_28)  
    …  
    train (Y_13, X_1), …, (Y_95, X_83), test (Y_96_hat, X_84), …,
    (Y_100_hat, X_88)  
2.  Select the model with the best performance on Y_31_hat, …,
    Y_100_hat  
3.  Retrain with (Y_13, X_1), …, (Y_100, X_88) and forecast Y_101_hat,
    …Y_105_hat given X_89, …, X_93

# Forecast examples

##### US Personal Consumption Expenditure

``` r
# 1. Load the data and plot 
library(fpp)
```

    ## Warning: package 'fpp' was built under R version 4.3.2

    ## Loading required package: forecast

    ## Warning: package 'forecast' was built under R version 4.3.2

    ## Registered S3 method overwritten by 'quantmod':
    ##   method            from
    ##   as.zoo.data.frame zoo

    ## Loading required package: fma

    ## Warning: package 'fma' was built under R version 4.3.2

    ## Loading required package: expsmooth

    ## Warning: package 'expsmooth' was built under R version 4.3.2

    ## Loading required package: lmtest

    ## Warning: package 'lmtest' was built under R version 4.3.2

    ## Loading required package: zoo

    ## 
    ## Attaching package: 'zoo'

    ## The following objects are masked from 'package:base':
    ## 
    ##     as.Date, as.Date.numeric

    ## Loading required package: tseries

    ## Warning: package 'tseries' was built under R version 4.3.2

``` r
library(forecast)
library(ggplot2)

data("usconsumption",package="fpp")
autoplot(usconsumption[,"consumption"]) + geom_point(shape=1,size=1)
```

![](6_forecasting_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->

``` r
# 2. Training testing split 
# reserve the last two years' data (i.e., 8 quarters) as the test set, while the rest being the training set
usconsumption_train = window(usconsumption[,"consumption"], start=c(1970,1), end=c(2008,4))
usconsumption_test = window(usconsumption[,"consumption"], start=c(2009,1), end=c(2010,4))
# window: a generic function which extracts the subset of the object x observed between the times start and end

# 3. Fit and train the model  
# we emphasize the forecasting part, and hence use a simple auto.arima() result
# in practice, select a model carefully (in MODEL SELECTION) before proceed to forecasting 
consump_fit = auto.arima(usconsumption_train, seasonal=F, d=0) # d is the order of first differencing 
consump_fit # ARIMA(1,0,1) with non-zero mean
```

    ## Series: usconsumption_train 
    ## ARIMA(1,0,1) with non-zero mean 
    ## 
    ## Coefficients:
    ##          ar1      ma1    mean
    ##       0.7776  -0.4912  0.7473
    ## s.e.  0.0962   0.1178  0.1163
    ## 
    ## sigma^2 = 0.4166:  log likelihood = -151.66
    ## AIC=311.33   AICc=311.59   BIC=323.53

``` r
# 4. Forecasting 
# forecast(): the function for forecasting, where h represents the forecast horizon
# since we want to forecast 8 quarters ahead, let h=8
consump_pred <- forecast(consump_fit, h=8) 

# 5. Check how the result looks like: 
# it consists of not only the forecast value, but lower/upper bounds for 80% and 95% Prediction Interval
consump_pred 
```

    ##         Point Forecast      Lo 80     Hi 80      Lo 95    Hi 95
    ## 2009 Q1    -0.18482819 -1.0120110 0.6423546 -1.4498951 1.080239
    ## 2009 Q2     0.02243512 -0.8380193 0.8828895 -1.2935163 1.338387
    ## 2009 Q3     0.18361236 -0.6963522 1.0635769 -1.1621773 1.529402
    ## 2009 Q4     0.30895103 -0.5826048 1.2005069 -1.0545660 1.672468
    ## 2010 Q1     0.40641999 -0.4920729 1.3049129 -0.9677064 1.780546
    ## 2010 Q2     0.48221623 -0.4204459 1.3848784 -0.8982864 1.862719
    ## 2010 Q3     0.54115879 -0.3640153 1.4463329 -0.8431855 1.925503
    ## 2010 Q4     0.58699517 -0.3196946 1.4936849 -0.7996671 1.973657

``` r
# 6. Plot the true series and the forecast series in the same figure
autoplot(consump_pred) +
  autolayer(usconsumption_test, series="Data") +
  autolayer(consump_pred$mean, series="Forecasts") +
  labs(title="US Personal Consumption Expenditure", y="Consumption")
```

![](6_forecasting_files/figure-gfm/unnamed-chunk-1-2.png)<!-- -->

##### CO2 levels at the Artic circle

``` r
data(co2, package = "TSA")
autoplot(co2) + geom_point(shape = 1, size = 1)
```

![](6_forecasting_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

``` r
# reserve the last two years' data as the test set
# since we have monthly data, two years mean 24 months ahead 
co2_train = window(co2,start=c(1994,1), end=c(2002,12))
co2_test = window(co2,start=c(2003,1), end=c(2004,12))

# the rest of the analysis is similar to the previous example
arima_est <- auto.arima(co2_train)
arima_est
```

    ## Series: co2_train 
    ## ARIMA(1,0,1)(0,1,1)[12] with drift 
    ## 
    ## Coefficients:
    ##          ar1      ma1     sma1   drift
    ##       0.8099  -0.4704  -0.8784  0.1463
    ## s.e.  0.1009   0.1408   0.1989  0.0054
    ## 
    ## sigma^2 = 0.4998:  log likelihood = -107.59
    ## AIC=225.19   AICc=225.86   BIC=238.01

``` r
arima_pred <- forecast(arima_est,h=24)
autoplot(arima_pred) +
  autolayer(co2_test, series="Data") +
  autolayer(arima_pred$mean, series="Forecasts") +
  labs(title="CO2 levels at the Artic circle", y="CO2 level")
```

![](6_forecasting_files/figure-gfm/unnamed-chunk-2-2.png)<!-- -->

##### Bitcoin cash price

``` r
bitcoin = read.csv("G:/My Drive/UMN Courses/6431/bitcoin_cash_price.csv", header=T)
bitcoin = as.data.frame(bitcoin)
bitcoin = bitcoin[213:1,] # reverse the index to make it into chronological order

# select the closing price
bit_close = ts(bitcoin[,5], start=1, end=213)
# select the volume and convert them into ts() format
bitcoin$Volume <- as.numeric(gsub(",","", bitcoin$Volume)) # to remove "," in volume
vol = ts(bitcoin[,6], start=1, end=213)
# plot the price 
plot(bit_close, type='o')
```

![](6_forecasting_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

Consider the closing price from 7/23/17 to 10/29/17 for training, and
forecast the future values in the next 10 days

``` r
bit_close_train = ts(bit_close[1:99], start=1, end=99)
bit_close_test = ts(bit_close[100:109],start=100,end=109)

arima_est <- auto.arima(bit_close_train)
arima_est
```

    ## Series: bit_close_train 
    ## ARIMA(2,1,0) 
    ## 
    ## Coefficients:
    ##          ar1      ar2
    ##       0.2654  -0.2953
    ## s.e.  0.0964   0.0964
    ## 
    ## sigma^2 = 2310:  log likelihood = -517.67
    ## AIC=1041.33   AICc=1041.59   BIC=1049.09

``` r
arima_pred <- forecast(arima_est, h=10)
autoplot(arima_pred) +
  autolayer(bit_close_test, series="Data") +
  autolayer(arima_pred$mean, series="Forecasts")
```

![](6_forecasting_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

``` r
# install a package to calculate the root mean square error
library(Metrics)
```

    ## Warning: package 'Metrics' was built under R version 4.3.2

    ## 
    ## Attaching package: 'Metrics'

    ## The following object is masked from 'package:forecast':
    ## 
    ##     accuracy

``` r
rmse(arima_pred$mean, bit_close_test) # 154.1558
```

    ## [1] 154.1558

Consider the closing price from 7/23/17 to 11/8/17 for training, and
forecast the future values in the next 10 days

``` r
# this time a great leap is included
# see how ARIMA reacts to it
bit_close_train = ts(bit_close[1:109], start=1, end=109)
bit_close_test = ts(bit_close[110:119], start=110, end=119)

arima_est <- auto.arima(bit_close_train)
arima_est
```

    ## Series: bit_close_train 
    ## ARIMA(1,0,1) with non-zero mean 
    ## 
    ## Coefficients:
    ##          ar1     ma1      mean
    ##       0.8726  0.3588  452.6314
    ## s.e.  0.0485  0.0921   44.7169
    ## 
    ## sigma^2 = 2219:  log likelihood = -574.11
    ## AIC=1156.21   AICc=1156.6   BIC=1166.98

``` r
arima_pred <- forecast(arima_est, h=10)
autoplot(arima_pred) +
  autolayer(bit_close_test, series="Data") +
  autolayer(arima_pred$mean, series="Forecasts") 
```

![](6_forecasting_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

``` r
rmse(arima_pred$mean,bit_close_test) # 659.7983
```

    ## [1] 659.7983

Now we consider the entire time series

``` r
# see if ARIMA performs better since the time series becomes longer
bit_close_train = ts(bit_close[1:203], start=1, end=203)
bit_close_test = ts(bit_close[204:213], start=204, end=213)

arima_est <- auto.arima(bit_close_train)
arima_est
```

    ## Series: bit_close_train 
    ## ARIMA(0,1,0) 
    ## 
    ## sigma^2 = 31080:  log likelihood = -1331.4
    ## AIC=2664.81   AICc=2664.83   BIC=2668.12

``` r
#auto.arima() believes that it's just a random walk!

arima_pred <- forecast(arima_est,h=10)
autoplot(arima_pred) +
  autolayer(bit_close_test, series="Data") +
  autolayer(arima_pred$mean, series="Forecasts")
```

![](6_forecasting_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

``` r
rmse(arima_pred$mean,bit_close_test) # 189.0594
```

    ## [1] 189.0594

``` r
# it seems to perform better given a longer time series 
```

Now we consider ARMAX and use Volume as an independent variable.

``` r
# recall the close price and the volume series below
bit_close = ts(bitcoin[,5], start=1, end=213)
vol = ts(bitcoin[,6], start=1, end=213)

# suppose we use Volume 10 days behind as auxiliary variables, how should we formulate it?
bit_close_train_reg = ts(bit_close[11:203], start=1, end=193)
bit_close_test_reg = ts(bit_close[204:213], start=194, end=203)
vol_train = ts(vol[1:193], start=1, end=193)
vol_test = ts(vol[194:203], start=194, end=203)

# fit the auto arima model
# Arima(bit_close_train,order=c(0,1,0),xreg=vol_train) if you do it manually
arima_x=auto.arima(bit_close_train_reg,xreg=vol_train) 
arima_x
```

    ## Series: bit_close_train_reg 
    ## Regression with ARIMA(0,1,2) errors 
    ## 
    ## Coefficients:
    ##          ma1      ma2   drift  xreg
    ##       0.0445  -0.2656  4.4914     0
    ## s.e.  0.0747   0.0761  0.1046     0
    ## 
    ## sigma^2 = 31141:  log likelihood = -1263.73
    ## AIC=2537.46   AICc=2537.79   BIC=2553.75

``` r
arima_pred <- forecast(arima_x,xreg = vol_test,h=10)
autoplot(arima_pred) +
  autolayer(bit_close_test_reg, series="Data") +
  autolayer(arima_pred$mean, series="Forecasts")
```

![](6_forecasting_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

``` r
rmse(arima_pred$mean,bit_close_test_reg) #174.9585
```

    ## [1] 174.9585

``` r
# it performs better with auxiliary variables than ARIMA alone
```

##### Supplementary Materials

The following are some simulation examples to calculate the forecasts
manually

AR(1)

``` r
# AR1
set.seed(888)
Y <- arima.sim(list(order = c(1,0,0), ar = 0.8), n = 100)
model1 = arima(Y, order=c(1,0,0), method="ML")
Yhat = predict(model1, n.ahead=20)$pred
ts.plot(c(Y,Yhat))
abline(v=100, lty=2, col="red"); abline(h=model1$coef[2], lty=2, col="blue")
```

![](6_forecasting_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

``` r
mu=model1$coef[2] # intercept, non-zero mean 
phi=model1$coef[1]
Y101=mu+phi*(Y[100]-mu) 

# show the prediction 
as.numeric(Y101) # calculated manually
```

    ## [1] 0.8993532

``` r
Yhat[1] # calculated by R
```

    ## [1] 0.8993532

MA(1)

``` r
set.seed(888)
X <- arima.sim(list(order = c(0,0,1), ma = -0.6), n = 100)
# ts.plot(X)
# acf(X)

model2 = arima(X, order=c(0,0,1), method="ML")
Xhat = predict(model2,n.ahead=8)$pred
ts.plot(c(X,Xhat))
abline(v=100, lty=2, col="red")
```

![](6_forecasting_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

``` r
theta = model2$coef[1]
mu = model2$coef[2] # intercept
X101 = mu+theta*model2$residuals[100]  #Calculated manually

# show the prediction 
as.numeric(X101) # calculated manually 
```

    ## [1] -0.4236135

``` r
Xhat[1] # calculated by R
```

    ## [1] -0.4236135

ARMA(1,1)

``` r
set.seed(666)
Z <- arima.sim(list(order = c(1,0,1), ar=0.8, ma = -0.6), n = 100)

# use auto.arima to fit automatically 
model3.0 = auto.arima(Z, d=0, seasonal = F)
model3.0 # auto.arima gives ARIMA(0,0,1) with zero mean 
```

    ## Series: Z 
    ## ARIMA(0,0,1) with zero mean 
    ## 
    ## Coefficients:
    ##          ma1
    ##       0.2469
    ## s.e.  0.1062
    ## 
    ## sigma^2 = 0.914:  log likelihood = -136.93
    ## AIC=277.85   AICc=277.97   BIC=283.06

``` r
library(forecast)
autoplot(forecast::forecast(model3.0, h=20)) # use the "forecast()" function from the package "forecast"
```

![](6_forecasting_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

``` r
# use arima to fit manually 
model3 = arima(Z, order=c(1,0,1), method="ML") 
model3 
```

    ## 
    ## Call:
    ## arima(x = Z, order = c(1, 0, 1), method = "ML")
    ## 
    ## Coefficients:
    ##           ar1     ma1  intercept
    ##       -0.5549  0.7671    -0.1398
    ## s.e.   0.3278  0.2757     0.1068
    ## 
    ## sigma^2 estimated as 0.8834:  log likelihood = -135.77,  aic = 279.54

``` r
Zhat = predict(model3,n.ahead=20)$pred
ts.plot(c(Z, Zhat)) # no gap between observed and predicted values
abline(v=100, lty=2, col="red")
```

![](6_forecasting_files/figure-gfm/unnamed-chunk-10-2.png)<!-- -->

``` r
mu=model3$coef[3]
phi=model3$coef[1]
theta=model3$coef[2]
Z101=mu+phi*(Z[100]-mu)+theta*model3$residuals[100]

# show the prediction 
as.numeric(Z101) # calculated manually
```

    ## [1] -0.1008919

``` r
Zhat[1] # calculated by R
```

    ## [1] -0.1008919
