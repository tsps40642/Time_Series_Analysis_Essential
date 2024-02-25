BASIC CONCEPTS
================

# Basic Statistics

### Expectation

1)  Estimated by y_bar = (Σ(i=1 to n) y_i)/n for observations y_1, …,
    y_n  
2)  E(X+Y) = E(X) + E(Y)  
3)  E(aY) = a\*E(Y), a = const.

### Variance

1)  Estimated by (Σ(i=i to n) (y_i-y_bar)<sup>2</sup>)/n  
2)  Var(X) = E\[(X-E(X))<sup>2</sup>\] = E(X<sup>2</sup>) -
    \[E(X)\]<sup>2</sup>
3)  Var(X+Y) = Var(X) + Var(Y), if X and Y are independent  
4)  Var(aX+b) = a<sup>2</sup>\*Var(X), a, b = const.

### Covariance & Correlation

1)  Cov(X,Y) = E\[(X−E(X))(Y−E(Y))\] = E(XY) - E(X)\*E(Y)  
2)  Cov(X, X) = Var(X)  
3)  Cov(aX+b, Y) = a<sup>2</sup>\*Cov(X, Y), transformation on Y the
    same  
4)  Cov(X1+X2, Y1+Y2) = Cov(X1, Y1) + Cov(X1, Y2) + Cov(X2, Y1) +
    Cov(X2, Y2)  
5)  Cov(X, Y) = 0 if X and Y are independent  
6)  Corr(X, Y) = Cov(X, Y)/sd(X)\*sd(Y)  
7)  Corr(aX+b, Y) = Corr(X, Y) ∀ ac\>0, i.e. invariance of correlation
    to linear transformation

# Frequently used terminology

### Predict vs. Forecast

1)  For predicting, observed values and predicted values may not have
    time order  
2)  For forecasting, it’s a subset of prediction task which has time
    order

### Attributes for time series

1)  Trend: refer to non-seasonal trend, like vehicle sales  
2)  Seasonality: usually known prior to time series analysis, like
    holiday season  
3)  Autocorr.:  
    1)  autocorr. is the corr. of a time series with a delayed copy of
        itself  
    2)  i.i.d. no longer hold \>\> default dependent  
    3)  First-order autocorr. = Corr(Y_t-1, Y_t)  
    4)  Second-order autocorr. = Corr(Y_t-2, Y_t), others the same  
    5)  Sometimes seasonality entails (first-order) autocorr. but not
        always the case (no causality)
4)  Variance: varying variance is called heteroscedasticity, opposite to
    homoscedasticity

### Stochastic process and sample paths

1.  Stochastic process: a sequence of r.v., {Y_t\|t = 1, 2, …}  
2.  Sample path: one realization of all r.v. in a given stochastic
    process, i.e. one possible/realized sequence of outcomes  
3.  One stochastic process can generate many sample paths

### The essence of time series analysis and forecasting

1)  we’re not only fitting lines to our data  
2)  we’re understanding possible sample paths that a stochastic process
    could take
