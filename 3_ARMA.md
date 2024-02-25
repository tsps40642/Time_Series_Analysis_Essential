ARMA
================

# ACF, autocorr. function (details in MODEL SELECTION)

DEF. Corr(Y_t, Y_t-k), k = 1, 2, 3, …, is a func. of time notation (t
and k)

# ARMA model

Given e_t ~ N(0, σ^2) is i.i.d. r.v.

## MA

1.  Moving average model  
2.  Models short-term impact, like campaigns, surge in demand

##### MA(1)

DEF. Y_t = e_t + θ*e_t-1, θ is the parameter to be estimated  
1. E(Y_t) = 0  
2. Cov(Y_t, Y_t-1) = θ\*σ<sup>2</sup> = const.,  
Cov(Y_t, Y_t-2) = Cov(Y_t, Y_t-3) = … = 0  
3. Var(Y_t) = (1+θ<sup>2</sup>)\*σ<sup>2</sup> = const.  
4. First-order corr: Corr(Y_t, Y_t-1) = Cov(Y_t, Y_t-1) /
\[sd(Y_t)\*sd(Y_t-1)\]  
= θ*σ<sup>2</sup> / (1+θ<sup>2</sup>)\*σ<sup>2</sup>  
= θ / (1+θ<sup>2</sup>)  
5. After the second-order corr.: Corr(Y_t, Y_t-2) = Cov(Y_t, Y_t-2) /
\[sd(Y_t)\*sd(Y_t-2) \]  
= 0 / (1+θ<sup>2</sup>)\*σ<sup>2</sup>  
= 0  
6. MA(1) model only has its first-order autocorr. being non-zore i.e.,
MA(1) assumes data at each point is only associated with its immediate
predecessor  
7. Note that none of the mean, covariance, all-order autocorr., variance
is a func. of t, i.e., all being const. over time, i.e., stationary

Plot ACF of MA(1)

``` r
# arima.sim: simulate from an ARIMA Model
# e.g. ts.sim <- arima.sim(list(order = c(1,1,0), ar = 0.7), n = 200) 

# Arima: fit ARIMA model to univariate time series

set.seed(666)
N = 500 # set the length of the time series to be 500
y1 = arima.sim(model=list(ma=0.8, sd=1), n=N) # use arima.sim() function to simulate a sample path of MA(1) of length N
acf(y1) # check the ACF of the simulated sample path
```

![](3_ARMA_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->

Fit with MA(1)

``` r
library(forecast)
```

    ## Warning: package 'forecast' was built under R version 4.3.2

    ## Registered S3 method overwritten by 'quantmod':
    ##   method            from
    ##   as.zoo.data.frame zoo

``` r
arima11 = Arima(y1, order=c(0,0,1)) # pretend that the data are real and that we don't know the true model; we fit with MA(1)
arima11 # check the results provided by MA(1)
```

    ## Series: y1 
    ## ARIMA(0,0,1) with non-zero mean 
    ## 
    ## Coefficients:
    ##          ma1     mean
    ##       0.8463  -0.0703
    ## s.e.  0.0289   0.0833
    ## 
    ## sigma^2 = 1.024:  log likelihood = -714.91
    ## AIC=1435.82   AICc=1435.87   BIC=1448.47

``` r
library(lmtest)
```

    ## Warning: package 'lmtest' was built under R version 4.3.2

    ## Loading required package: zoo

    ## 
    ## Attaching package: 'zoo'

    ## The following objects are masked from 'package:base':
    ## 
    ##     as.Date, as.Date.numeric

``` r
coeftest(arima11) 
```

    ## 
    ## z test of coefficients:
    ## 
    ##            Estimate Std. Error z value Pr(>|z|)    
    ## ma1        0.846256   0.028876 29.3063   <2e-16 ***
    ## intercept -0.070327   0.083289 -0.8444   0.3985    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
# a function provided by the package "lmtest", which returns the significance of coefficients as in the "summary()" of a linear model
```

##### MA(2)

DEF. Y_t = e_t + θ1*e_t-1 + θ2*e_t-2, θ1, θ2 are the parameters to be
estimated  
1. E(Y_t) = 0  
2. Var(Y_t) = (1+θ1<sup>2</sup>+θ2<sup>2</sup>)*σ<sup>2</sup> = const.  
3. Corr(Y_t, Y_t-1) = (θ1+θ1*θ2) / (1+θ1<sup>2</sup>+θ2<sup>2</sup>) =
const. != 0  
Corr(Y_t, Y_t-2) = θ2 / (1+θ1<sup>2</sup>+θ2<sup>2</sup>) = const. !=
0  
Corr(Y_t, Y_t-3) = 0, …  
4. After the second-order, Corr. = 0

Fit with MA(2) (it’s actually mis-specified)

``` r
# we pretend that the data are real and that we don't know the true model; we fit with MA(2)
arima12 = Arima(y1,order=c(0,0,2)) # note that y1 is actually having ma=0.8     
arima12 # check the results provided by MA(2)
```

    ## Series: y1 
    ## ARIMA(0,0,2) with non-zero mean 
    ## 
    ## Coefficients:
    ##          ma1      ma2     mean
    ##       0.8267  -0.0271  -0.0707
    ## s.e.  0.0435   0.0459   0.0812
    ## 
    ## sigma^2 = 1.025:  log likelihood = -714.74
    ## AIC=1437.47   AICc=1437.56   BIC=1454.33

``` r
coeftest(arima12)
```

    ## 
    ## z test of coefficients:
    ## 
    ##            Estimate Std. Error z value Pr(>|z|)    
    ## ma1        0.826706   0.043456 19.0241   <2e-16 ***
    ## ma2       -0.027084   0.045945 -0.5895   0.5555    
    ## intercept -0.070747   0.081163 -0.8717   0.3834    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
# since y1 is actually having ma=0.8, the MA(2) model is mis-specified for this time series, thus the coeff. of second-order is not significant 
```

##### MA(q)

DEF. Y_t = e_t + θ1*e_t-1 + θ2*e_t-2 + … + θq\*e_t-q, θt(t=1 to q) are
the parameters to be estimated  
1. E(Y_t) = 0  
2. After the qth-order, autocorr. = 0

##### MA(q) when mean of the time != 0

DEF. Y_t = μ + e_t + θ1*e_t-1 + θ2*e_t-2 + … + θq\*e_t-q, μ, θt(t=1 to
q) are the parameters to be estimated

1.  E(Y_t) = μ = const., which is usually unknown in real data and can
    be estimated with θ’s simultaneously  
2.  After the qth-order, autocorr. = 0

## AR

1.  Auto regressive model  
2.  Models long-term (but gradually reduced) impact  
3.  AR model is one of the first invented model to fit the time series
    data

##### AR(1)

DEF. Y_t = φ*Y_t-1 + e_t, φ is some constant  
1. Y_t = φ*Y_t-1 + e_t = φ*(φ*Y_t-2 + e_t-1) + e_t = φ*(φ*(φ\*Y_t-3 +
e_t-2) + e_t-1) + e_t = …, Y_t is associated with any Y_t-k  
2. Need abs(φ) \< 1 to guarantee stationarity since when φ = 1 it
becomes random walk which is not stationary. In standard software
package, we don’t usually fit models with abs(φ) \> 1  
3. Autocorr. tails off in ACF, converging to 0 but never reach exact 0  
(compared to MA(q), which autocorr. = 0 after qth-order)

``` r
# ACF of AR(1)
y2 = arima.sim(model=list(ar=0.8, sd=1), n=N)
Acf(y2)
```

![](3_ARMA_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

``` r
Acf(y2, lag.max = 50)                     
```

![](3_ARMA_files/figure-gfm/unnamed-chunk-4-2.png)<!-- -->

``` r
# lag.max=50 plots a longer ACF, which is particularly helpful when we suspect there're seasonal effects
# However, we should note that the longer the lag, the fewer the data points are available for calculating the autocorrelation (what does professor mean for this???)
```

##### AR(2)

DEF. Y_t = φ1*Y_t-1 + φ2*Y_t-2 + e_t, φ1, φ2 are some constants  
1. Autocorr. tails off, converging to 0 but never reach exact 0, but ACF
of AR(2) is similar to AR(1)’s, so can’t use ACF to distinguish them.  
2. We should use PACF (P for partial) (introduced in later chapter)

##### AR(p)

DEF. Y_t = φ1*Y_t-1 + φ2*Y_t-2 + … + φp\*Y_t-p + e_t, φt(t=1 to p) are
some constants  
1. We can consider AR model as a linear regression of Y_t (the response
) against Y_t-1, … Y_t-p (exploratory variables)  
2. Modern machine learning models like XGBoost and LSTM consider Y_t =
f(Y_t-1, … Y_t-p) + e_t, and usually the answer of this would be
satisfying

##### AR(p) when mean of the time != 0

DEF. Y_t-μ = φ1*(Y_t-1-μ) + φ2*(Y_t-2-μ) + … + φp\*(Y_t-p-μ) + e_t,
φt(t=1 to p) are some constants  
1. Define Y_t_tilde = Y_t-μ, and E(Y_t_tilde) = 0 is the original form

##### A fun example of AR

``` r
# comparing Y_t = Y_{t-1} + e_t with Y_t = 2*Y_{t-1} + e_t
e = rnorm(N) # generating white noise 
y3 = cumsum(e) # a regular random walk

# creating Y_t = 2*Y_{t-1} + e_t through a for loop
y4 = rep(0,N) 
y4[1] = e[1] 
for (t in 2:N){ # from 2 
  y4[t] = 2 * y4[t-1] + e[t]
}
par(mfrow = c(2,1))
plot(y3, type='o', pch = 20, cex = 0.5)
plot(y4, type='o', pch = 20, cex = 0.5)
```

![](3_ARMA_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

``` r
# note that the ranges are so different 
```

##### Backshift operator

DEF. B is a backshift operator if B*Y_t = Y_t-1  
1. B is a func. symbol without an actual value, like time machine  
2. B can be applied multiple times: (B^2)*Y_t = B*(B*Y_t) = B*Y_t-1 =
Y_t-2  
3. Rewrite AR model in B:  
(1) AR(1): Y_t = φ*Y_t-1 + e_t = φ*(B*Y_t) + e_t, so e_t = Y_t*(1-φ*B)  
(2) AR(2): Y_t = φ1*Y_t-1 + φ2*Y_t-2 + e_t = φ1*(B*Y_t) +
φ2*(B<sup>2</sup>*Y_t) + e_t, so e_t = Y_t*(1-φ1*B-φ2*B<sup>2</sup>)  
(3) AR(p): e_t = Y_t*(1 - φ1*B - φ2*B<sup>2</sup> - … -
φp*B<sup>p</sup>) = Y_t*Φ(x),  
Φ(x) = (1 - φ1*B - φ2*B<sup>2</sup> - … - φp\*B<sup>p</sup>) is the
characteristic func. of AR(p)

##### Characteristic funcion

DEF. Φ(x) = (1 - φ1*B - φ2*B<sup>2</sup> - … - φp\*B<sup>p</sup>) is the
characteristic poly. of AR(p)  
1. The roots of this func. describe the behavior of the give AR
process  
2. AR(1) is stationary if Φ(x)=0 has ALL roots abs(x) \> 1, which meets
abs(φ) \< 1 as mentioned before  
3. But “which meets abs(φ) \< 1” can’t be generalized to AR(p), since
each term of AR(p) may still \> 1 in the equation  
4. To find roots of a given poly, use polyroot() in R  
5. (other details in MODEL SELECTION)

## Compare the sample path of MA(1) and the sample path of AR(1)

``` r
# y1 = arima.sim(model=list(ma=0.8, sd=1), n=N)
# y2 = arima.sim(model=list(ar=0.8, sd=1), n=N)

# plot y_t
par(mfrow=c(2,1)) 
plot(y1, type='o', ylim=c(-4,4), pch = 20, cex = 0.5) #MA(1)
plot(y2, type='o', ylim=c(-4,4), pch = 20, cex = 0.5) #AR(1)
```

![](3_ARMA_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

Check first-order autocorr.

``` r
# plot y_t against y_{t-1} (corresponding to the first-order autocorrelation)
par(mfrow=c(2,1))
plot(y1[1:(N-1)], y1[2:N], pch = 20, cex = 0.5)
plot(y2[1:(N-1)], y2[2:N], pch = 20, cex = 0.5)
```

![](3_ARMA_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

``` r
# calculate the first-order autocorrelation 
cor(y1[1:(N-1)], y1[2:N]) 
```

    ## [1] 0.4967902

``` r
cor(y2[1:(N-1)], y2[2:N]) 
```

    ## [1] 0.7634159

Check second-order autocorr.

``` r
#Plot y_t against y_{t-2} (corresponding to the second-order autocorrelation)
par(mfrow=c(2,1))
plot(y1[1:(N-2)], y1[3:N], pch = 20, cex = 0.5)
plot(y2[1:(N-2)], y2[3:N], pch = 20, cex = 0.5)
```

![](3_ARMA_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

``` r
# calculate the second-order autocorrelation 
cor(y1[1:(N-2)], y1[3:N]) 
```

    ## [1] -0.01346394

``` r
cor(y2[1:(N-2)], y2[3:N]) 
```

    ## [1] 0.6056953

## ARMA

Combine AR and MA models to capture long-term and short-term memory
respectively

##### ARMA(1, 1)

DEF. Y_t = φ1*Y_t-1 + e_t + θ1*e_t-1 = AR part + new info. + MA part

##### ARMA(p, q)

DEF. Y_t = φ1*Y_t-1 + φ2*Y_t-2 + … + φp*Y_t-p + e_t + θ1*e_t-1 +
θ2*e_t-2 + … + θq*e_t-q  
= AR(p) + (new information at time t) + MA(q)

1.  Note that since MA(q) is always stationary, ARMA(p, q) is stationary
    if AR(p) part is stationary  
2.  We can use ACF to identify orders of MA(q), but can’t for AR(p) and
    ARNA(p, q)  
3.  Based on Wold’s theorem, ARMA(p, q) is the best linear approximation
    to any arbitrary stationary process, and t’s the Taylor’s expansion
    of time series  
4.  Business advantages:
    1)  fast
    2)  low labor or maintenance cost
    3)  robustness

### Wold’s theorem (FYI)

In statistics, Wold’s decomposition or the Wold representation theorem
says that every covariance-stationary time series can be written as the
sum of two time series, one deterministic and one stochastic

# Simulation practice

Suppose we want to simulate a sample path from AR(2) with phi1=0.8 and
phi2=0.6, but we are not sure if this series is stationary. Then check
with the Characteristic Polynomial by filling in the code below:

``` r
polyroot(c(1, -0.8, -0.6)) 
```

    ## [1]  0.7862996+0i -2.1196330+0i

``` r
# entering opposite-signed coefficients that corresponds to the char. polynomial of the AR(2) above
# e.g. Y_t = 0.8*Y_t-1 + 0.6*Y_t-2 + e_t >> polyroot(c(1, -0.8, -0.6))

# let's see if we can generate a sample path of the AR(2) above regardless of stationarity
# will have error in arima.sim(model = list(ar = c(0.8, 0.6), sd = 1), n = 100) since 'ar' part of model is not stationary 
```

If we want to generate a sample path from a non-stationary process,
should use the code below

``` r
set.seed(42)
N <- 100L # 100 sample paths
T <- 20L # each sample path has a length of 20 time stamps i.e. length=20 

df2c <- data.frame(Y = rep(NA, N * T), id = rep(NA, N * T), t = rep(NA, N * T))

for (i in 1:N) { # i: 
  # Initialize the process for each path
  Y <- numeric(T) # generate 0 as initialization 
  Y[1:3] <- rnorm(3)  # initial values for the first third time stamps 

  for (t in 4:T) { # T: time stamps start from 4 
    # Generate the stochastic process for the new process
    Y[t] <- 2.4 * Y[t - 1] - 1.55 * Y[t - 2] + 0.3 * Y[t - 3] + rnorm(1)
  }

  # Assign values to the data frame
  start_idx <- (i - 1) * T + 1
  end_idx <- i * T
  df2c$Y[start_idx:end_idx] <- Y
  df2c$id[start_idx:end_idx] <- rep(i, T)
  df2c$t[start_idx:end_idx] <- 1:T
}
```
