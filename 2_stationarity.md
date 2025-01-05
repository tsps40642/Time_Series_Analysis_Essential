STATIONARITY
================

# Meaning and definition of stationarity

## Why stationarity is important?

Trend, seasonality, autocorr., and variance are features we try to
isolate for a given time series data. They are features about the data,
not about the model, and hence we need to figure them out to regardless
of what models we’re going to use.

Stationarity, however, guarantees that statistical properties don’t
change over time, and hence if we have stationarity, a bunch of models
become applicable, and we have more tools to analyse the data.

So basically, stationarity results easier modeling and forecasting, it
simplifies the complexities within time series data, making it easier to
model and forecast than non-stationary time series. Stationaryity allows
us to discover the full process from one sample path.

## Stationarity process requires

1)  mean has no trend i.e. mean = const.
2)  has no varying autocorr. ∀ order i.e. autocorr. = const.
3)  has no varying variance i.e. variance = const.
4)  stationarity allows seasonality  
    So stationarity only exclude non-const. mean, autocorr., and
    variance

## A process Y is stationarity if

1)  const. mean: E(Y_t) = const.  
2)  const. autocov.: Cov(Y_t, Y_t-k) = const.
    1)  if time lag = k, autocov. is a func. of k, not a func of t  
    2)  kth-order autocorr. = Corr(Y_t, Y_t-k) = Cov(Y_t,
        Y_t-k)/sd(Y_t)\*sd(Y_t-k) = const.  
3)  implied from (2), Var(Y_t) = Cov(Y_t, Y_t) = const.
    1)  implied from (2) and (3), Corr(Y_t, Y_t-k) = Cov(Y_t,
        Y_t-k)/Var(Y_t) = const.

# Identify stationay process

Given e_t ~ N(0, σ<sup>2</sup>) is i.i.d. r.v.

## White noise is stationary

### DEF. Y_t = e_t

### Attributes

1)  no trend  
2)  no seasonality  
3)  no autocorr.  
4)  variance = const.  

### Statistics

1)  E(Y_t) = 0 = const.  
2)  Cov(Y_t, Y_t-1) = 0  
3)  Var(Y_t) = σ<sup>2</sup> = const.

### Importance of white noise 
1)  It’s the fundamental building block in modeling more complex
    stochastic processes  
2)  In real business world, it’s a part of the variation beyond human control like:  
    a)  systematic or measurement error  
    b)  New info. that the model can’t capture (an interpretation almost
        exclusive to time series)  
    c)  Experiments controlled by a higher dimension  

``` r
e = rnorm(100) # generate white noise time series 
ts.plot(e) # plot the data in order ("ts" stands for time series)
```

![](2_stationarity_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->

``` r
plot(e[1:99], e[2:100]) # plot the first-order correlation
```

![](2_stationarity_files/figure-gfm/unnamed-chunk-1-2.png)<!-- -->

``` r
cor(e[1:99], e[2:100]) # calculate the first-order autocorrelation
```

    ## [1] -0.129316

## Moving average is stationary

### DEF. MA(1): Y_t = e_t + θ\*e_t-1, θ is a parameter to be estimated

(The moving average carries information from the previous time point)

### Attributes

1)  no trend  
2)  no seasonality  
3)  some degree of autocorr.  
4)  variance = const.

### Statistics for MA(1)

1)  E(Y_t) = 0  
2)  Cov(Y_t, Y_t-1) = θ\*σ<sup>2</sup> = const.,  
    Cov(Y_t, Y_t-2) = Cov(Y_t, Y_t-3) = … = 0  
3)  Var(Y_t) = (1+θ<sup>2</sup>)\*σ<sup>2</sup> = const.
4)  First-order corr: Corr(Y_t, Y_t-1) = Cov(Y_t, Y_t-1) /
    \[sd(Y_t)\*sd(Y_t-1)\]  
    = θ*σ<sup>2</sup> / (1+θ<sup>2</sup>)*σ<sup>2</sup>  
    = θ / (1+θ<sup>2</sup>)  
5)  After the first-order corr.: Corr(Y_t, Y_t-2) = Cov(Y_t, Y_t-2) /
    \[sd(Y_t)\*sd(Y_t-2) \]  
    = 0 / (1+θ<sup>2</sup>)\*σ<sup>2</sup>  
    = 0  
6)  MA(1) model only has its first-order autocorr. being non-zore i.e.,
    MA(1) assumes data at each point is only associated with its
    immediate predecessor  
7)  Note that none of the mean, covariance, all-order autocorr.,
    variance is a func. of t, i.e., all being const. over time, i.e.,
    stationary

### Moving Average, θ = 1

``` r
e = rnorm(100) # generate white noise time series 
Y = ts((e+1*zlag(e))) # i.e. Y_t = e_t + e_{t-1}
# zlag() function returns the immediate predecessor of each element

plot(Y, type='o') # plot the moving average in a different way, same as the ts.plot(Y)
```

![](2_stationarity_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

``` r
plot(Y[1:99], Y[2:100]) # plot consecutive observations to check the first-order correlation
abline(a=0, b=1, lwd=2, col="red") # superimpose y=x, to see if there's a trend 
```

![](2_stationarity_files/figure-gfm/unnamed-chunk-2-2.png)<!-- -->

``` r
cor(Y[2:99], Y[3:100]) # calculate the first-order autocorrelation 
```

    ## [1] 0.47419

``` r
# Y[1] becomes NA due to no predecessor for e[1], so our calculation of correlation starts from Y[2]
```

### Moving Average, θ = 0.4

``` r
e = rnorm(100)
Y = ts(e+0.4*zlag(e)) # this gives us Y_t = e_t + 0.4*e_{t-1}

plot(Y, type='o') # plot the moving average in a different way, same as the ts.plot(Y)
```

![](2_stationarity_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

``` r
plot(Y[2:99], Y[3:100]) # expect to see weaker correlation than the previous moving average
```

![](2_stationarity_files/figure-gfm/unnamed-chunk-3-2.png)<!-- -->

``` r
cor(Y[2:99], Y[3:100]) # calculate the first-order autocorrelation
```

    ## [1] 0.3312874

## Random walk is NOT stationary

### DEF. Y_t = Σ(i=1 to t)e_i, i.e. Y_t = Y_t-1 + e_t

### Attributes

1)  no trend  
2)  no seasonality  
3)  strong autocorr.  
4)  variance != const.

### Statistics

1)  E(Y_t) = 0  
2)  Var(Y_1) = Var(e_1) = σ<sup>2</sup>,  
    Var(Y_2) = Var(e_1+e_2) = 2*σ<sup>2</sup>, …  
    Var(Y_t) = t*σ<sup>2</sup> != const., varying with t  

``` r
e = rnorm(100) # generate white noise time series 
y = cumsum(e) # calculate the cumulative sums, this is how we generate random walks in R
ts.plot(y) # plot the random walk
```

![](2_stationarity_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

``` r
plot(y[1:99], y[2:100]) # expect to see much stronger correlation than that in the MAs
```

![](2_stationarity_files/figure-gfm/unnamed-chunk-4-2.png)<!-- -->

``` r
cor(y[1:99], y[2:100]) # didn't use zlag() here. So we can use y[1] 
```

    ## [1] 0.9479042

``` r
# there will be clear autocorr. 
```

## Other examples for sample paths: increase variance over time

``` r
e = rnorm(100) # generate White Noise again
t = 1:100 # generate an index t, ranging from 1 to 100
Y = ts(t*e) # let the "swing" of Y to be multiplied by t. What will Y look like? 
plot(Y, type='o') # plot the time series. Do you expect the autocorrelation to be high or low? 
```

![](2_stationarity_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

``` r
plot(Y[1:99],Y[2:100]) # check the autocorrelation
abline(a=0, b=1, lwd=2, col="red")
```

![](2_stationarity_files/figure-gfm/unnamed-chunk-5-2.png)<!-- -->

``` r
cor(Y[1:99], Y[2:100])
```

    ## [1] 0.1011868

``` r
# autocorrelation should be low since it has nothing to do with its predecessor at each t 
# variance increase over time since it's multiplied by t 
```

## Other examples for sample paths: spurious autocorr.

``` r
Y = ts(t+e) # let there be a constant t imposed over time. What will Y look like?
plot(Y, type='o') # plot the time series. Do you expect the autocorrelation to be high or low?
```

![](2_stationarity_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

``` r
plot(Y[1:99], Y[2:100]) # check the autocorrelation
abline(a=0, b=1, lwd=2, col="red")
```

![](2_stationarity_files/figure-gfm/unnamed-chunk-6-2.png)<!-- -->

``` r
cor(Y[1:99], Y[2:100])  
```

    ## [1] 0.9987968

``` r
# the autocorrelation above is spurious  
# the true one should be calculated after removing the trend (i.e., the mean of each t) 

# the correct practice to fix the problem of spurious antocorr.: 
# remove time-variant components i.e. remove its trend i.e. deduct t from time series before plotting
plot(Y[1:99]-c(1:99), Y[2:100]-c(2:100)) # check the autocorrelation after removing the trend E(Y)=t
abline(a=0, b=1, lwd=2, col="red")
```

![](2_stationarity_files/figure-gfm/unnamed-chunk-6-3.png)<!-- -->

``` r
cor(Y[1:99]-c(1:99), Y[2:100]-c(2:100)) # the true autocorrelation is very weak 
```

    ## [1] 0.1293772
