MODEL SELECTION
================

In this chapter, will discuss multiple criteria for model selection

# ACF (details)

1.  DEF. Corr(Y_t, Y_t-k), k = 1, 2, 3, …, is auto corr. func., is a
    func. of time notation (t and k)  
2.  ACF can help us distinguish autocorr. order in MA(q), but can’t for
    AR(p) since it will tailing off  
3.  ACF CAN’T:
    1)  determine the order of AR(p)  
    2)  separate ARMA from AR processes  
    3)  determine the order of ARMA process  
    4)  separate ARMA from ARIMA processes

# PACF

DEF. Corr(Y_t - Y_t_hat, Y_t-k - Y_t-k_hat) is partial autocorr. func.
for a stationary stochastic process Y_t  
It is the leftover correlation between Y_t and Y_t-k while holding
Y_t-1, …, Y_t-k+1 const.

### Motivation for PACF: conditionaing on something

Consider an AR(1) process: Y_t = φ*Y_t-1 + e_t, φ = 0.8, given Y_t-1 =
13,  
Then Y_t\|Y_t-1 = 0.8*13 + e_t, that is, conditioning on Y_t-1, it
degenerate into white noise.  
That is, condition on something means something is realization, no
longer r.v.

ACF and PACF for AR(2)

``` r
library(TSA) # for zlag
```

    ## Warning: package 'TSA' was built under R version 4.3.2

    ## 
    ## Attaching package: 'TSA'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     acf, arima

    ## The following object is masked from 'package:utils':
    ## 
    ##     tar

``` r
library(forecast) # for auto.arima
```

    ## Warning: package 'forecast' was built under R version 4.3.2

    ## Registered S3 method overwritten by 'quantmod':
    ##   method            from
    ##   as.zoo.data.frame zoo

    ## Registered S3 methods overwritten by 'forecast':
    ##   method       from
    ##   fitted.Arima TSA 
    ##   plot.Arima   TSA

``` r
library(ggplot2) # for coeftest 

set.seed(88)
y = arima.sim(model=list(order=c(2,0,0), ar=c(0.5,0.4), sd=1), n=1000) # simulate AR(2) data
par(mfrow=c(1,2))
acf(y, main = "ACF") # check the ACF: tails off
Pacf(y, main = "PACF") # check the PACF: only first and second order are significant 
```

![](5_model_selection_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->

ACF and PACF for MA(2)

``` r
set.seed(88)
y = arima.sim(model=list(order=c(0,0,2), ma=c(0.9,0.9), sd=1), n=1000) # simulate MA(2) data
par(mfrow=c(1,2))
acf(y, main = "ACF") # check the ACF: only first and second order are significant 
Pacf(y, main = "PACF") # check the PACF: tails off 
```

![](5_model_selection_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

# EACF

The idea is given a fixed p and remove AR(p) to find the smallues q s.t.
MA(q) looks nice on ACF  
1. The result of ACF and PACF are reliable yet it’s heuristic for eacf
(i.e. only suggestion rather than decisive)  
2. Sometimes is not the correct one, one should use it to select
candidate models rather than the final decision  
3. Eacf won’t show seasonality of the time series

ACF and PACF for ARMA model will bith tail off, so we use eacf

``` r
set.seed(88)
y = arima.sim(model=list(order=c(2,0,2), ar=c(0.5,0.4), ma=c(0.9,0.9), sd=1), n=1000)
par(mfrow=c(1,2))
acf(y, main = "ACF") # check the ACF: tails off 
Pacf(y, main = "PACF") # check the PACF: tails off 
```

![](5_model_selection_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

``` r
# since neither ACF nor PACF can help us identify the order of ARMA, we use eacf
eacf(y) 
```

    ## AR/MA
    ##   0 1 2 3 4 5 6 7 8 9 10 11 12 13
    ## 0 x x x x x x x x x x x  x  x  x 
    ## 1 x x x o o o o o o o o  o  o  o 
    ## 2 x x o o o o o o o o o  o  o  o 
    ## 3 x x x o x o o o o o o  o  o  o 
    ## 4 x x x x x o o o o o o  o  o  o 
    ## 5 x x x x o x o o o o o  o  o  o 
    ## 6 x x x x x o o o o o o  o  o  o 
    ## 7 x x x o x o x o o o o  o  o  o

``` r
# in this example, EACF provides two candidate models: ARMA(2,2) and ARMA(1,3)
```

# ADF (augmented Dickey-Fuller test, or unit root test)

If ADF report p-value \< specified α, it means not containing random
walk component yet not necessarily stationary

### Unit root

DEF. The process has a unit root if root z=1 is a root of its
characteristic poly

##### Example: test the presence of unit root for AR(1): Y_t = φ\*Y_t-1 + e_t

Subtract Y_t-1 from both LHS and RHS: Y_t-Y_t-1 = (φ-1)\*Y_t-1 + e_t,
then if φ=1 the time series is non-stationary  
(we do the first-order diff to the series, and if the underlying process
contains a random walk component, after doing the diff. RHS becomes
white noise, that’s the idea of ADF test)

H0: φ=1 i.e. there’s a unit root in a given AR model, implied
non-stationary  
H1: φ\<1 i.e. stationary

``` r
# ADF test for stationarity
library(tseries)
```

    ## Warning: package 'tseries' was built under R version 4.3.2

``` r
y <- arima.sim(list(order = c(1,0,0), ar = 0.2), n = 1000) # generate AR(1) with phi=0.2 
adf.test(y) # stationary 
```

    ## Warning in adf.test(y): p-value smaller than printed p-value

    ## 
    ##  Augmented Dickey-Fuller Test
    ## 
    ## data:  y
    ## Dickey-Fuller = -9.9428, Lag order = 9, p-value = 0.01
    ## alternative hypothesis: stationary

``` r
# also use polyroot to check stationary 
polyroot(c(1, -0.2)) # all roots>1, stationary 
```

    ## [1] 5+0i

``` r
# AR(1) with phi=0.95, very close to a random walk
y <- arima.sim(list(order = c(1,0,0), ar = 0.95), n = 1000)
adf.test(y) # stationary 
```

    ## Warning in adf.test(y): p-value smaller than printed p-value

    ## 
    ##  Augmented Dickey-Fuller Test
    ## 
    ## data:  y
    ## Dickey-Fuller = -5.1876, Lag order = 9, p-value = 0.01
    ## alternative hypothesis: stationary

``` r
# also use polyroot to check stationary 
polyroot(c(1, -0.95)) # all roots>1, stationary 
```

    ## [1] 1.052632+0i

``` r
# A real random walk
y = cumsum(rnorm(1000))
adf.test(y) # non-stationary 
```

    ## 
    ##  Augmented Dickey-Fuller Test
    ## 
    ## data:  y
    ## Dickey-Fuller = -2.4321, Lag order = 9, p-value = 0.3954
    ## alternative hypothesis: stationary

##### Example

In this example, polyroot() and adf.test() doesn’t coordinate, might due
to the simulation paths is too short or should simulate more times

``` r
# 1. 
# consider Y_t = 0.8*Y_t-1 + 0.6*Y_t-2 + e_t 
# we do polyroot() to check stationarity 
polyroot(c(1, -0.8, -0.6)) # non-stationary 
```

    ## [1]  0.7862996+0i -2.1196330+0i

``` r
# 2. 
# so we can't use arima.sim() since it will cause error 
# should simulate manually 
set.seed(42)
N <- 1000L # 100 sample paths
T <- 20L # each sample path has a length of 20 time stamps i.e. length=20 
df2c <- data.frame(Y = rep(NA, N * T), id = rep(NA, N * T), t = rep(NA, N * T))

for (i in 1:N) { # i: 
  # Initialize the process for each path 
  Y <- numeric(T) # generate 0 as initialization 
  Y[1:2] <- rnorm(2)  # initial values for the first third time stamps 

  for (t in 3:T) { # T: time stamps start from 4 
    # Generate the stochastic process for the new process
    Y[t] <- 0.8*Y[t - 1] +0.6*Y[t - 2] + rnorm(1) 
  }

  # Assign values to the data frame
  start_idx <- (i - 1) * T + 1
  end_idx <- i * T
  df2c$Y[start_idx:end_idx] <- Y
  df2c$id[start_idx:end_idx] <- rep(i, T)
  df2c$t[start_idx:end_idx] <- 1:T
}

# 3. 
# do adf.test and find the result is not consistent
# lengthen the sample path or simulate more times 
adf.test(df2c$Y)
```

    ## Warning in adf.test(df2c$Y): p-value smaller than printed p-value

    ## 
    ##  Augmented Dickey-Fuller Test
    ## 
    ## data:  df2c$Y
    ## Dickey-Fuller = -24.294, Lag order = 27, p-value = 0.01
    ## alternative hypothesis: stationary

# AIC, BIC

### Attributes of AIC and BIC

1.  Both AIC and BIC the lower the better  
2.  AIC
    1.  Preferred when choosing bigger model for goodness of fit,
        accuracy  
    2.  As n approaches to infinite, AIC might pick more and more
        complex model since it consider goodness of fit  
3.  BIC:
    1.  Preferred when choosing smaller model for transparency, compact,
        interpretability, and eaiser for communication, since it
        penalizes the extra model parameters more heavily then AIC  
    2.  As n approaches to infinite, BIC would tend to converge to a
        specific model in the model sets  
4.  AIC and BIC not always agree with each other

### Limitations of AIC and BIC

1.  AIC, BIC are universal, so it’s legitimate to compare AR(1)’s AIC
    with ARMA(10, 10)’s AIC  
2.  AIC, BIC are not specially designed for time series, their results
    in time series analysis are rough. As long as the data can generate
    MLE, we can use these criteria  
3.  Based on 2., ACF, PACF are specially designed for time series
    analysis, so they’re more reliable and should be prioritized when
    analyzing time series, like the standard procedure below

# The practical procedure for time series analysis

### Example: US Personal Consumption Expenditure

(is already a stationary time series)

``` r
library(fpp)
```

    ## Warning: package 'fpp' was built under R version 4.3.2

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

``` r
data("usconsumption", package="fpp")
Y = usconsumption[,"consumption"]
```

Steps 1. to 3. are reliable since they’re designed for time series
analysis

``` r
# 1. Visualize the data
plot(Y) + geom_point(shape=1, size=1) 
```

![](5_model_selection_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

    ## NULL

``` r
# 2. ADF test to check stationarity. If it is not stationary, consider: 
# (1) differencing and/or seasonal differencing, or
# (2) transformation, or 
# (3) regression to remove the trend (which we haven't learned yet), 
# and then apply the ADF test to check stationarity again
adf.test(Y) 
```

    ## Warning in adf.test(Y): p-value smaller than printed p-value

    ## 
    ##  Augmented Dickey-Fuller Test
    ## 
    ## data:  Y
    ## Dickey-Fuller = -4.2556, Lag order = 5, p-value = 0.01
    ## alternative hypothesis: stationary

``` r
# 3. Plot ACF and PACF to check if we are lucky to get a pure AR or MA model 
Acf(Y) # 3 bars are above significance line
```

![](5_model_selection_files/figure-gfm/unnamed-chunk-7-2.png)<!-- -->

``` r
Pacf(Y) # 3 bars are above significance line
```

![](5_model_selection_files/figure-gfm/unnamed-chunk-7-3.png)<!-- -->

Steps 4. to 7. are used for generate candidate models and do selection

``` r
# 4. If ARMA model has to be considered, use EACF to identify a region in which the true model may be located
eacf(Y) # eacf suggests ARMA(1, 1) 
```

    ## AR/MA
    ##   0 1 2 3 4 5 6 7 8 9 10 11 12 13
    ## 0 x x x o o o o o o o o  o  o  o 
    ## 1 x o x o o o o o o o o  o  o  o 
    ## 2 x o x o o o o x o o o  o  o  o 
    ## 3 x x o o o o o x o o o  o  o  o 
    ## 4 x o o o o o o x o o o  o  o  o 
    ## 5 x x o o x o o o o o o  o  o  o 
    ## 6 x o x o x o o o o o o  o  o  o 
    ## 7 x o x x x o o o o o o  o  o  x

``` r
# 5. Try a few candidate models, compare their AIC and BIC
# selecting from candidate models may depends on domain knowledge 
Arima(Y,order=c(3,0,0)) # AR(3), only long-term effect. AIC=318.16, BIC=333.66
```

    ## Series: Y 
    ## ARIMA(3,0,0) with non-zero mean 
    ## 
    ## Coefficients:
    ##          ar1     ar2     ar3    mean
    ##       0.2366  0.1603  0.1909  0.7533
    ## s.e.  0.0763  0.0774  0.0759  0.1153
    ## 
    ## sigma^2 = 0.3921:  log likelihood = -154.08
    ## AIC=318.16   AICc=318.54   BIC=333.66

``` r
Arima(Y,order=c(0,0,3)) # MA(3), only long-term effect. Based on domain knowledge at this point, this is more preferred. AIC=319.46, BIC=334.96
```

    ## Series: Y 
    ## ARIMA(0,0,3) with non-zero mean 
    ## 
    ## Coefficients:
    ##          ma1     ma2     ma3    mean
    ##       0.2542  0.2260  0.2695  0.7562
    ## s.e.  0.0767  0.0779  0.0692  0.0844
    ## 
    ## sigma^2 = 0.3953:  log likelihood = -154.73
    ## AIC=319.46   AICc=319.84   BIC=334.96

``` r
Arima(Y,order=c(1,0,1)) # ARMA(1, 1) from eacf, AIC=320.62, BIC=333.02
```

    ## Series: Y 
    ## ARIMA(1,0,1) with non-zero mean 
    ## 
    ## Coefficients:
    ##          ar1      ma1    mean
    ##       0.7724  -0.4814  0.7528
    ## s.e.  0.0846   0.1086  0.1099
    ## 
    ## sigma^2 = 0.4006:  log likelihood = -156.31
    ## AIC=320.62   AICc=320.87   BIC=333.02

``` r
# note that the final decision base on AIC and BIC is different in this example 

# 6. Also consider the model selected by auto.arima()
# note that shouldn't use auto.arima() to make the final decision, rather it should be used to generate candidate models 
consump_fit = auto.arima(Y, seasonal=F, d=0) 
consump_fit # ARIMA(0,0,3). AIC=319.46, BIC=334.96
```

    ## Series: Y 
    ## ARIMA(0,0,3) with non-zero mean 
    ## 
    ## Coefficients:
    ##          ma1     ma2     ma3    mean
    ##       0.2542  0.2260  0.2695  0.7562
    ## s.e.  0.0767  0.0779  0.0692  0.0844
    ## 
    ## sigma^2 = 0.3953:  log likelihood = -154.73
    ## AIC=319.46   AICc=319.84   BIC=334.96

``` r
# 7. If all models' AICs and BICs are close, choose one based on domain knowledge or with the highest forecast accuracy

# 8. Not all data scientists agree on the same model because of domain knowledge and preference, which is fine
```

Suppliment: The following considers a linear model

``` r
lm1 = lm(Y~c(1:164)) 
summary(lm1)
```

    ## 
    ## Call:
    ## lm(formula = Y ~ c(1:164))
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -3.13511 -0.31637  0.04457  0.39459  1.41686 
    ## 
    ## Coefficients:
    ##              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)  0.924853   0.108069   8.558 8.37e-15 ***
    ## c(1:164)    -0.002055   0.001136  -1.809   0.0724 .  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.6888 on 162 degrees of freedom
    ## Multiple R-squared:  0.01979,    Adjusted R-squared:  0.01374 
    ## F-statistic: 3.271 on 1 and 162 DF,  p-value: 0.07237

``` r
AIC(lm1) # can compare AIC of a time series model with AIC of a linear model
```

    ## [1] 347.1286

### Example: Bitcoin cash price

(non-stationary, need to take first-order differencing)

``` r
bitcoin = read.csv("G:/My Drive/UMN Courses/6431/bitcoin_cash_price.csv", header=T)
bitcoin = as.data.frame(bitcoin)
# consider the closing price from 7/23/17 to 10/29/17; change to the chronological order
bit_close = ts(bitcoin[213:115,5], start=1, end=99)
```

``` r
# 1. Plot
plot(bit_close) + geom_point(shape = 1, size = 1)
```

![](5_model_selection_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

    ## NULL

``` r
# 2. ADF test to check stationarity. If it is not stationary, consider: 
# (1) differencing and/or seasonal differencing, or
# (2) transformation, or 
# (3) regression to remove the trend (which we haven't learned yet), 
# and then apply the ADF test to check stationarity again
adf.test(bit_close) # has random walk component of non-stationarity 
```

    ## 
    ##  Augmented Dickey-Fuller Test
    ## 
    ## data:  bit_close
    ## Dickey-Fuller = -1.8676, Lag order = 4, p-value = 0.6314
    ## alternative hypothesis: stationary

``` r
# thus we do differencing then do ADF test again 
bit_diff1 = diff(bit_close, differences = 1)
adf.test(bit_diff1) # non-stationarity is gone after the 1st-order differencing
```

    ## Warning in adf.test(bit_diff1): p-value smaller than printed p-value

    ## 
    ##  Augmented Dickey-Fuller Test
    ## 
    ## data:  bit_diff1
    ## Dickey-Fuller = -4.4666, Lag order = 4, p-value = 0.01
    ## alternative hypothesis: stationary

``` r
# 3. Plot ACF, PACF, and eacf(if needed) to check if we are lucky to get a pure AR or MA model 
Acf(bit_diff1) # suggest no AR and MA order 
```

![](5_model_selection_files/figure-gfm/unnamed-chunk-11-2.png)<!-- -->

``` r
Pacf(bit_diff1) # suggest ARIMA(2, 1, 0), with seasonality 13 
```

![](5_model_selection_files/figure-gfm/unnamed-chunk-11-3.png)<!-- -->

``` r
eacf(bit_diff1) # suggest ARIMA(2, 1, 1)
```

    ## AR/MA
    ##   0 1 2 3 4 5 6 7 8 9 10 11 12 13
    ## 0 x x x o o o o o o o o  o  o  o 
    ## 1 x x x o o o o o o o o  o  o  o 
    ## 2 x o o o o o o o o o o  o  x  o 
    ## 3 x o o o o o o o o o o  o  x  o 
    ## 4 x o o o o o o o o o o  o  o  o 
    ## 5 x o o o o o o o o o o  o  o  o 
    ## 6 x x x o o o o o o o o  o  o  o 
    ## 7 x x o o o o o o o o o  o  o  o

``` r
# 5. Try a few candidate models, compare their AIC and BIC
Arima(bit_close, order=c(0,1,0)) # if we follow ACF and consider signals in PACF as false positive, then we may end up with this model
```

    ## Series: bit_close 
    ## ARIMA(0,1,0) 
    ## 
    ## sigma^2 = 2593:  log likelihood = -524.23
    ## AIC=1050.46   AICc=1050.51   BIC=1053.05

``` r
Arima(bit_close, order=c(2,1,0), seasonal=list(order=c(1,0,0), period=13)) # if we follow PACF and consider no signals in ACF as false negative, then we may end up with this one. This one has the best AIC and BIC
```

    ## Series: bit_close 
    ## ARIMA(2,1,0)(1,0,0)[13] 
    ## 
    ## Coefficients:
    ##          ar1      ar2     sar1
    ##       0.2763  -0.3174  -0.2385
    ## s.e.  0.0957   0.0957   0.1000
    ## 
    ## sigma^2 = 2191:  log likelihood = -514.96
    ## AIC=1037.92   AICc=1038.35   BIC=1048.26

``` r
Arima(bit_close, order=c(2,1,0)) # we get this one (or ARIMA(0,1,3)) if we follow EACF
```

    ## Series: bit_close 
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
# 6. Also consider the model selected by auto.arima() 
arima_est <- auto.arima(bit_close)
arima_est #auto.arima agrees with EACF
```

    ## Series: bit_close 
    ## ARIMA(2,1,0) 
    ## 
    ## Coefficients:
    ##          ar1      ar2
    ##       0.2654  -0.2953
    ## s.e.  0.0964   0.0964
    ## 
    ## sigma^2 = 2310:  log likelihood = -517.67
    ## AIC=1041.33   AICc=1041.59   BIC=1049.09

### Example: CO2 levels at the Artic circle

(non-stationary, need to take second-order differencing )

``` r
data(co2, package = "TSA")

# 1. Plot
plot(co2) + geom_point(shape = 1, size = 1)
```

![](5_model_selection_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->

    ## NULL

``` r
# 2. ADF test 
adf.test(co2) 
```

    ## Warning in adf.test(co2): p-value smaller than printed p-value

    ## 
    ##  Augmented Dickey-Fuller Test
    ## 
    ## data:  co2
    ## Dickey-Fuller = -9.9788, Lag order = 5, p-value = 0.01
    ## alternative hypothesis: stationary

``` r
# In practice, ADF test entails automatic de-trending
# In other words, it can't detect non-stationarity caused by a deterministic trend
# In this particular example, we still proceed with non-stationarity even though ADF test is passed, because of the clear trend
# co2_diff1 = ts(lm(as.vector(co2)~c(1:132))$residuals) #This commented line allows us to remove the linear trend as a deterministic trend, which is not necessarily optimal in this particular example, but will be learned in general in Lecture 8 

# (1) first we consider seasonal differencing: Y_t - Y_{t-12}, following the result of "auto.arima(co2)" -- ARIMA(1,0,1)(0,1,1)[12]
co2_diff1 = diff(co2,differences = 1, lag = 12) 
plot(co2_diff1) + geom_point(shape = 1, size = 1)
```

![](5_model_selection_files/figure-gfm/unnamed-chunk-12-2.png)<!-- -->

    ## NULL

``` r
adf.test(co2_diff1)
```

    ## 
    ##  Augmented Dickey-Fuller Test
    ## 
    ## data:  co2_diff1
    ## Dickey-Fuller = -3.0091, Lag order = 4, p-value = 0.1575
    ## alternative hypothesis: stationary

``` r
# (2) Then we consider Z_t - Z_{t-1}, where Z_t = Y_t - Y_{t-12}
co2_diff2=diff(co2_diff1, lag=1) 
plot(co2_diff2) + geom_point(shape = 1, size = 1) # looks stationary now 
```

![](5_model_selection_files/figure-gfm/unnamed-chunk-12-3.png)<!-- -->

    ## NULL

``` r
adf.test(co2_diff2)
```

    ## Warning in adf.test(co2_diff2): p-value smaller than printed p-value

    ## 
    ##  Augmented Dickey-Fuller Test
    ## 
    ## data:  co2_diff2
    ## Dickey-Fuller = -5.9106, Lag order = 4, p-value = 0.01
    ## alternative hypothesis: stationary

``` r
# Overall this is a try-and-error process, so you don't necessarily need to follow my route above
# Please feel free to try multiple combinations (w./w.o. regular/seasonal differencing, or even try a 2nd-order differencing)

# 3. Check ACF and PACF, allowing 5 cycles; ACF shows Seasonal MA (P=0 and Q=1), but p and q are still open to discussion
Acf(co2_diff2,lag.max =61) 
```

![](5_model_selection_files/figure-gfm/unnamed-chunk-12-4.png)<!-- -->

``` r
Pacf(co2_diff2,lag.max = 61)
```

![](5_model_selection_files/figure-gfm/unnamed-chunk-12-5.png)<!-- -->

``` r
eacf(co2_diff2) 
```

    ## AR/MA
    ##   0 1 2 3 4 5 6 7 8 9 10 11 12 13
    ## 0 x o o o o o o o o o x  x  x  o 
    ## 1 x x o o o o o o o o o  x  x  o 
    ## 2 o o o o o o o o o o o  x  o  x 
    ## 3 o o x o o o o o o o o  x  o  x 
    ## 4 o o x o o o o o o o o  x  o  x 
    ## 5 x o x o o o o o o o o  x  o  o 
    ## 6 o o x o o o o o o o o  x  o  o 
    ## 7 o x x o o o o o o o o  o  o  o

``` r
Arima(co2,order=c(0,1,1), seasonal=list(order=c(0,1,1), period=12))
```

    ## Series: co2 
    ## ARIMA(0,1,1)(0,1,1)[12] 
    ## 
    ## Coefficients:
    ##           ma1     sma1
    ##       -0.5792  -0.8206
    ## s.e.   0.0791   0.1137
    ## 
    ## sigma^2 = 0.5683:  log likelihood = -139.54
    ## AIC=285.08   AICc=285.29   BIC=293.41

``` r
# also try auto.arima 
arima_est <- auto.arima(co2) 
arima_est # drift: y_t - a - b*t, which is the mean after differencing
```

    ## Series: co2 
    ## ARIMA(1,0,1)(0,1,1)[12] with drift 
    ## 
    ## Coefficients:
    ##          ar1      ma1     sma1   drift
    ##       0.8349  -0.4630  -0.8487  0.1520
    ## s.e.  0.0819   0.1246   0.1274  0.0052
    ## 
    ## sigma^2 = 0.5288:  log likelihood = -136.09
    ## AIC=282.18   AICc=282.7   BIC=296.11

``` r
arima_est2 <- auto.arima(co2, d=1, D=1) # can specify that regular and seasonal differencing are both needed, then this becomes exactly the same model we manually select above 
arima_est2
```

    ## Series: co2 
    ## ARIMA(0,1,1)(0,1,1)[12] 
    ## 
    ## Coefficients:
    ##           ma1     sma1
    ##       -0.5792  -0.8206
    ## s.e.   0.0791   0.1137
    ## 
    ## sigma^2 = 0.5683:  log likelihood = -139.54
    ## AIC=285.08   AICc=285.29   BIC=293.41

``` r
plot(co2, type='o', main="CO2 levels at the Artic circle", ylab="CO2 level")
lines(arima_est2$fitted, col="red")
```

![](5_model_selection_files/figure-gfm/unnamed-chunk-12-6.png)<!-- -->
