ARIMA and SARIMA
================

# Approaches to achieve stationarity

## Differencing

DEF. ΔY_t = Y_t - Y_t-1 a first difference operator  
1. Note that Δ = 1-B where B is a backshift operator  
2. Δ<sup>2</sup> = (1-B)<sup>2</sup> is a second-difference operator,
(Y_t-Y_t-1) - (Y_t-1-Y_t-2) = (1-B)<sup>2</sup>\*Y_t  
3. Differencing can remove both random walk components and
(deterministic) trends, thus converts non-stationary time series and
series with poly. trend into stationary ones  
4. But note that there’s cost:  
(1) for random walk: differencing effectively removes non-stationarity,
while fitting regression would cause false positive  
(2) (deterministic) trend: differencing would disturb our understanding
of the data thus lose prediction accuracy, while fitting regression
would effectively capture the fixed signal

##### Example: differencing and check ACF

More about how to select a proper model in be in MODEL SELECTION

``` r
library(forecast) # for auto.arima
```

    ## Warning: package 'forecast' was built under R version 4.3.2

    ## Registered S3 method overwritten by 'quantmod':
    ##   method            from
    ##   as.zoo.data.frame zoo

``` r
library(lmtest) # for coeftest
```

    ## Warning: package 'lmtest' was built under R version 4.3.2

    ## Loading required package: zoo

    ## 
    ## Attaching package: 'zoo'

    ## The following objects are masked from 'package:base':
    ## 
    ##     as.Date, as.Date.numeric

``` r
bitcoin = read.csv("G:/My Drive/UMN Courses/6431/bitcoin_cash_price.csv", header=T)
bitcoin = as.data.frame(bitcoin)
#consider the closing price from 7/23/17 to 10/29/17; change to the chronological order
bit_close = ts(bitcoin[213:115,5], start=1, end=99) # remember to specify start and end 

# first check the ACF and plot 
acf(bit_close) # autocorr. tails off 
```

![](4_ARIMA_and_SARIMA_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->

``` r
library(ggplot2)
plot(bit_close) + geom_point(shape = 1, size = 1)
```

![](4_ARIMA_and_SARIMA_files/figure-gfm/unnamed-chunk-1-2.png)<!-- -->

    ## NULL

``` r
# calculate the 1st-order difference of the time series
bit_diff1 = diff(bit_close) # default differences = 1
# check the ACF after 1st differencing
acf(bit_diff1) # there seems to be some patterns 
```

![](4_ARIMA_and_SARIMA_files/figure-gfm/unnamed-chunk-1-3.png)<!-- -->

``` r
#plot the 1st-order difference of the time series
plot(bit_diff1) + geom_point(shape = 1, size = 1)
```

![](4_ARIMA_and_SARIMA_files/figure-gfm/unnamed-chunk-1-4.png)<!-- -->

    ## NULL

``` r
# calculate the 2nd-order difference of the time series
bit_diff2 = diff(bit_close, differences = 2)
# check the ACF after 2nd differencing
acf(bit_diff2)
```

![](4_ARIMA_and_SARIMA_files/figure-gfm/unnamed-chunk-1-5.png)<!-- -->

``` r
#plot the 2nd-order difference of the time series
plot(bit_diff2) + geom_point(shape = 1, size = 1)
```

![](4_ARIMA_and_SARIMA_files/figure-gfm/unnamed-chunk-1-6.png)<!-- -->

    ## NULL

``` r
# fit with auto.arima()
# auto.arima sometimes is not the best but give you some hints 
bitcoin_fit <- auto.arima(bit_close)
bitcoin_fit
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
coeftest(bitcoin_fit)
```

    ## 
    ## z test of coefficients:
    ## 
    ##      Estimate Std. Error z value Pr(>|z|)   
    ## ar1  0.265434   0.096402  2.7534 0.005898 **
    ## ar2 -0.295256   0.096394 -3.0630 0.002191 **
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
# plot the fitted line (red) and the raw series
plot(bit_close, type='o')
lines(bitcoin_fit$fitted, col="red") #time lag between real events and estimated events
```

![](4_ARIMA_and_SARIMA_files/figure-gfm/unnamed-chunk-1-7.png)<!-- -->

``` r
par(mfrow = c(1,2))
# compare the result with AR(1)
AR1 = Arima(bit_close, order=c(1,0,0))
# plot the fitted line (red) and the raw series
plot(bit_close, type='o')
lines(AR1$fitted, col="red") 
coeftest(AR1)
```

    ## 
    ## z test of coefficients:
    ## 
    ##             Estimate Std. Error z value  Pr(>|z|)    
    ## ar1         0.907230   0.038627 23.4872 < 2.2e-16 ***
    ## intercept 432.501858  49.106053  8.8075 < 2.2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
# compare the result with AR(2)
AR2 = Arima(bit_close, order=c(2,0,0))
# plot the fitted line (red) and the raw series
plot(bit_close, type='o')
lines(AR2$fitted, col="red")
```

![](4_ARIMA_and_SARIMA_files/figure-gfm/unnamed-chunk-1-8.png)<!-- -->

``` r
coeftest(AR2)
```

    ## 
    ## z test of coefficients:
    ## 
    ##             Estimate Std. Error z value  Pr(>|z|)    
    ## ar1         1.142217   0.096452 11.8423 < 2.2e-16 ***
    ## ar2        -0.254398   0.096341 -2.6406  0.008276 ** 
    ## intercept 432.238102  40.664106 10.6295 < 2.2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
# both AR1 and AR2 has significant coeff, how to select a better one will be in MODEL SELECTION 
```

## De-trend: achieve stationarity through differencing

DEF. Y_t = t + e_t where t is purely a deterministic time index and will
have its unit  
1. Y_t is NOT stationary since E(Y_t) = t is a func. of t, not const.  
2. by differencing, ΔY_t = t+e_t - \[(t-1)+e_t-1\] = 1+e_t-e_t-1 is a
stationary MA(1)  
3. Practice procedure for difference operator  
(1) If we speculate the data has a linear trend and want to use
difference operator, then difference raw data  
(2) If no more linear trend then fit MA(1), if there’s still linear
trend then difference again until trend is removed  
(3) Conduct inference or forecasting  
(4) Once done, integrate forecast values back to the original scale

### Generalize to d-th order poly. trend

In general the process with d-th order poly. trend: Y_t = p_d(t) + e_t
becomes stationary after d-th order differencing Δ<sup>d</sup>\*Y_t is a
MA(d) model

##### Example: Y_t with dth order poly.trend becomes stationary after differencing d times

if d=2 then Y_t = t<sup>2</sup> + e_t, after differencing twice:  
1. let Z_t = ΔY_t = (t<sup>2</sup>-e_t) - \[(t-1)<sup>2</sup>+e_t-1\] =
2t-1+e_t-e_t-1, still has linear trend  
2. let W_t = ΔZ_t = Δ<sup>2</sup>Y_t = (2t-1+e_t-e_t-1) -
\[2(t-1)-1+e_t-1-e_t-2\] = 2+(e_t-e_t-1)-(e_t-1-e_t-2), no linear trend
\>\> end up having a stationary process after the 2nd differencing

##### Example: why differencing a time series with deterministic trend would reduce prediction accuracy?

let Y_t = t + e_t has a deterministic trend, then we know at t+2: Y_t+2
= (t+2) + e_t+2  
Yet if we do differencing then we acknowledge there’s random walk
component (which actually no), and by doing so will harm the prediction
accuracy, as shown below.  
1. Treat it as deterministic trend: E(Y_t+2) = E\[(t+2) + e_t+2\] = t+2,
which is the true expected value  
2. Treat it as random walk and do differencing:  
Y_t+1 = Y_t + e_t+1 \>\> Y_t+1_hat = E(Y_t + e_t+1) = Y_t, use Y_t+1_hat
to predict Y_t+2_hat  
Y_t+2 = Y_t+1 + e_t+2 \>\> E(Y_t+2) = E(Y_t+1 + e_t+2) = Y_t+1_hat =
Y_t  
So, the deterministic trend can’t be correctly captured thus the
prediction accuracy reduced

## ARIMA(p, d, q)

DEF. Y_t follows ARIMA(p, d, q) if Δ<sup>d</sup>\*Y_t follows ARMA(p,
q)  
Actually when software fit the ARIMA model, it will first take dth order
diff. then use diff.ed data to fit an ARMA(p, q) model

``` r
# simulated ARIMA(2,1,2)
# can see the range of the time series is very wide, although the standard deviation is set as 1
# can also see the pattern of the time series differs each time: up, down, or fluctuated
# however, the ACF remains high all the time, significantly
y = arima.sim(model=list(order=c(2,1,2), ar=c(0.9,0.09), ma=c(0.2,0.1), sd=1), n=1000)
par(mfrow=c(1,2))
ts.plot(y)
Acf(y)
```

![](4_ARIMA_and_SARIMA_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

##### Example of standard model fitting procedure: US Personal Consumption Expenditure

``` r
library(fpp) # install the package that contains the `US Consumption` data
```

    ## Warning: package 'fpp' was built under R version 4.3.2

    ## Loading required package: fma

    ## Warning: package 'fma' was built under R version 4.3.2

    ## Loading required package: expsmooth

    ## Warning: package 'expsmooth' was built under R version 4.3.2

    ## Loading required package: tseries

    ## Warning: package 'tseries' was built under R version 4.3.2

``` r
data("usconsumption", package="fpp") 
# load the data from the package
# it contains two columns: Percentage changes in quarterly personal consumption expenditure and personal disposable income for the US, 1970 to 2010 

# plot the time series
library(ggplot2)
Y = usconsumption[, "consumption"]
autoplot(Y) + geom_point(shape=1, size=1)
```

![](4_ARIMA_and_SARIMA_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

``` r
# check the ACF
par(mfrow=c(1,1))
Acf(Y)
```

![](4_ARIMA_and_SARIMA_files/figure-gfm/unnamed-chunk-3-2.png)<!-- -->

``` r
# fit with auto.arima()
# in auto.arima(), we can specify `seasonal = T` to enforce seasonality, 
# and/or `d=1` to enforce the 1st-order difference
# and/or `D=1` to enforce the 1st-order seasonal difference
consump_fit = auto.arima(Y, seasonal = F) #, d=0, D=1, seasonal = T)
consump_fit
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
coeftest(consump_fit) # the dependence of the first 3 orders are significant 
```

    ## 
    ## z test of coefficients:
    ## 
    ##           Estimate Std. Error z value  Pr(>|z|)    
    ## ma1       0.254208   0.076657  3.3162 0.0009126 ***
    ## ma2       0.226026   0.077871  2.9026 0.0037010 ** 
    ## ma3       0.269499   0.069221  3.8933 9.887e-05 ***
    ## intercept 0.756174   0.084405  8.9588 < 2.2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
# auto.arima gives MA(3)

# plot the fitted line (red) and the raw series
plot(Y, type='o', ylab="Consumption", main="US Personal Consumption Expenditure, auto fit")
lines(consump_fit$fitted, col="red")
```

![](4_ARIMA_and_SARIMA_files/figure-gfm/unnamed-chunk-3-3.png)<!-- -->

``` r
# MA(3) is the best reliable model so far (not globally best), but there's still some domain knowledge that you need to know in the industry 

# now let's fit a model following your intuition, say AR(3), ARMA(3,3), or whichever model you like
consump_fit_manual = Arima(Y, order=c(3,0,3))
consump_fit_manual
```

    ## Series: Y 
    ## ARIMA(3,0,3) with non-zero mean 
    ## 
    ## Coefficients:
    ##          ar1     ar2      ar3      ma1      ma2     ma3    mean
    ##       0.5487  0.4810  -0.4132  -0.2950  -0.4055  0.4796  0.7545
    ## s.e.  0.3340  0.2159   0.2513   0.3181   0.2074  0.1650  0.0968
    ## 
    ## sigma^2 = 0.3939:  log likelihood = -152.96
    ## AIC=321.92   AICc=322.84   BIC=346.71

``` r
coeftest(consump_fit_manual)
```

    ## 
    ## z test of coefficients:
    ## 
    ##            Estimate Std. Error z value  Pr(>|z|)    
    ## ar1        0.548731   0.333989  1.6430  0.100391    
    ## ar2        0.481045   0.215914  2.2279  0.025884 *  
    ## ar3       -0.413187   0.251306 -1.6442  0.100143    
    ## ma1       -0.294950   0.318057 -0.9274  0.353744    
    ## ma2       -0.405527   0.207374 -1.9555  0.050520 .  
    ## ma3        0.479634   0.165000  2.9069  0.003651 ** 
    ## intercept  0.754525   0.096804  7.7943 6.476e-15 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
# plot the fitted line (red) and the raw series
plot(Y, type='o', ylab="Consumption", main="US Personal Consumption Expenditure, manual fit")
lines(consump_fit_manual$fitted, col="blue")
```

![](4_ARIMA_and_SARIMA_files/figure-gfm/unnamed-chunk-3-4.png)<!-- -->

``` r
# we can repeat this process multiple times, if we have more than 1 candidate models
```

# SRIMA

## Seasonality

DEF. Y_t = φY_t-12 + e_t is a seasonal AR(1) or SAR(1)  
DEF. Y_t = φ1*Y_t-12 + φ2*Y_t-24 + e_t is a seasonal AR(2) or SAR(2)

Seasonality can be recognized by 1. observing raw data by eyes 2.
running ACF, cycle(ts) 3. domain knowledge

##### Simupation example: use e=norm() then specify y = e+0.8\*zlag(e, d=seasonality)

``` r
# check the ACF of a seasonal MA(1), with a cycle of 12
# that is, Y_t = e_t + 0.8*e_{t-12}
# arima.sim() does not support SARIMA data generation
library(TSA) # for zlag
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
e = rnorm(1000)
y = e+0.8*zlag(e, d=12) # specify lag=12
par(mfrow=c(1,2))
ts.plot(y) # visually can't see any seasonality
Acf(y) # but the ACF shows a clear seasonal pattern
```

![](4_ARIMA_and_SARIMA_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

``` r
par(mfrow=c(1,1))
# non-seasonal part 
cor(y[13:999],y[14:1000]) # the first-order autocorrelation is actually very weak
```

    ## [1] 0.04599756

``` r
plot(y[13:999],y[14:1000]) # start from t=13 since the first 12 data points are not available due to lag
```

![](4_ARIMA_and_SARIMA_files/figure-gfm/unnamed-chunk-4-2.png)<!-- -->

``` r
# seasonal part
cor(y[13:988],y[25:1000]) # the seasonal autocorrelation (lag=12) is strong, though
```

    ## [1] 0.5232298

``` r
plot(y[13:988],y[25:1000])
```

![](4_ARIMA_and_SARIMA_files/figure-gfm/unnamed-chunk-4-3.png)<!-- -->

##### Simupation example: use regular arima.sim

``` r
# check the ACF of a seasonal AR(1), with a cycle of 7
# that is, Y_t = 0.9*Y_{t-7} + e_t
# but assuming the first 6 AR coefficients as zero (coded as rep(0,6))
y = arima.sim(model=list(order=c(7,0,0), ar=c(rep(0,6),0.9), sd=1), n=1000)
par(mfrow=c(1,2))
ts.plot(y[1:50]) # seasonal pattern is visible, but each season is not identical
Acf(y, lag=50) # clear seasonal pattern (decay, rather than cut-off)
```

![](4_ARIMA_and_SARIMA_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

### Backshift operator in seasonal AR

DEF. with B, Y_t = φ*B<sup>12</sup>*Y_t + e_t thus
(1-φ*B<sup>12</sup>)*Y_t = e_t, suppose we have both AR(1) and SAR(1),  
then (1-φ1*B)*(1-Φ1*B<sup>12</sup>)*Y_t = e_t, that is, Y_t = φ1*Y_t-1 +
Φ1*Y_t-12 + φ1Φ1\*Y_t-13 + e_t

## Multiplicative seasonal ARIMA

DEF. ARIMA(p, d, q)(P, D, Q)\[S\] where S is the cycle length  
(refer to the formula on course slides)

##### Practice: specify the model

1.  Y_t = φ1*Y_t-1 + e_t + θ*e_t-1 + Θ\*e_t-12 \>\>\> ARIMA(1, 0, 1)(0,
    0, 1)\[12\]  
2.  Y_t = Θ1*Y_t-7 + Θ2*Y_t-14 + e_t + θ*e_t-1 + θ2*e_t-2 + Ψ\*e_t-7
    \>\>\> ARIMA(0, 0, 2)(2, 0, 1)\[7\]

##### Example:

## Transformation

1.  If observing increasing variance, can take log() first to see if it
    can be addressed  
2.  after 1. if there’s a linear trend, can fit the linear model to see
    if trend and be removed  
3.  after 2. if trend can’t be removed, then need other transformation  
4.  Practice rule of thumb of transformation: make sure the
    transformation is reversible
    1)  be careful of transformations that are NOT 1-1 func.  
    2)  taking subset like deleting, averaging which are irreversible,
        will lose information  
    3)  repeated use of data may increase bias

### BoxCox transformation

DEF. BoxCos transformation of Y_t into Z_t with a λ is:  
Z_t = log(Y_t), if λ = 0  
= (Y_t(<sup>λ</sup>)-1) / λ, o.w.  
λ is a tuning parameter to determine how concave/convex we want to
transformed data to be

### Calender adjustment

Since each month have different number of days, we normalize data by the
number of days

## Other notes

### Coeff. in random walk

If abs(coeff.) \< 1: historical information would be discounted before
adding to the current  
If abs(coeff.) \>= 1: the historical information will sum up  
It’s like the gate in the LSTM model

### The procedure for finding d in ARIMA(p, d, q) s.t. time series is stationary

1.  Do first-order differencing
    1)  check the stationarity of Y_t  
    2)  if no \>\> second-order  
    3)  repeat the above two steps until reach stationary
2.  fit the d-th order Y_t into ARMA(p, q) is equivalent to fit Y_t into
    ARIMA(p, d, q), but the reason that we do ARIMA is because we want
    the data is analyzed in the raw data form, so that we don’t have to
    transform it back to the original

### Differenced data in ARIMA

In ARIMA, the differenced data will be ensured to be stationary and will
be integrate back to Y_t (not difference yet)

### Priority

1.  Always check stationarity, if it is,  
2.  Knowing long term or short term (i.e. AR or MA)  
3.  Knowing the order of each of them (i.e. AR(2) or AR(3), MA(1 or
    MA(2))  
4.  Knowing how strong the dependency is (i.e. coeff.)
