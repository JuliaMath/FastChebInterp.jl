# FastChebInterp

[![Build Status](https://travis-ci.org/stevengj/FastChebInterp.jl.svg?branch=master)](https://travis-ci.org/stevengj/FastChebInterp.jl)

Fast multidimensional Chebyshev interpolation on a hypercube (Cartesian-product)
domain, using a separable (tensor-product) grid of Chebyshev interpolation points.

For domain upper and lower bounds `lb` and `ub`, and a given `order`
tuple, you would create an interpolator for a function `f` via:
```
using FastChebInterp
x = chebpoints(order, lb, ub) # an array of StaticVector
c = chebinterp(f.(x), lb, ub)
```
and then evaluate the interpolant for a point `y` (a vector)
via `c(y)`.

This package is an experimental replacement for some of the functionality in [ChebyshevApprox.jl](https://github.com/RJDennis/ChebyshevApprox.jl) in order to get more performance.  The [ApproxFun.jl](https://github.com/JuliaApproximation/ApproxFun.jl) package also performs Chebyshev interpolation and many other tasks.
