# FastChebInterp

[![Build Status](https://travis-ci.org/stevengj/FastChebInterp.jl.svg?branch=master)](https://travis-ci.org/stevengj/FastChebInterp.jl)

Fast multidimensional Chebyshev interpolation on a hypercube (Cartesian-product)
domain, using a separable (tensor-product) grid of Chebyshev interpolation points.

(Note: this package is currently unregistered, so you have to supply the full URL when installing it in Julia. Type `add https://github.com/stevengj/FastChebInterp.jl` at the `pkg>` prompt.)

For domain upper and lower bounds `lb` and `ub`, and a given `order`
tuple, you would create an interpolator for a function `f` via:
```
using FastChebInterp
x = chebpoints(order, lb, ub) # an array of StaticVector
c = chebfit(f.(x), lb, ub)
```
and then evaluate the interpolant for a point `y` (a vector)
via `c(y)`.

We also provide a function `chebgradient(c,y)` that returns a tuple `(c(y), ∇c(y))` of
the interpolant and its gradient at a point `y`.

The FastChebInterp package also supports complex and vector-valued functions `f`.  In
this case, `c(y)` returns a vector of interpolants, and one can use `chebjacobian(c,y)`
to compute the tuple `(c(y), J(y))` of the interpolant and its Jacobian matrix at `y`.

We also have a function `chebregression(x, y, [lb, ub,], order)` that
can perform multidimensional Chebyshev least-square fitting.  It
returns a Chebyshev polynomial of a given `order` (tuple) fit
to a set of points `x[i]` and values `y[i]`, optionally in a box
with bounds `lb, ub` (which default to bounding box for `x`).

This package is an experimental replacement for some of the functionality in [ChebyshevApprox.jl](https://github.com/RJDennis/ChebyshevApprox.jl) in order to get more performance.  The [ApproxFun.jl](https://github.com/JuliaApproximation/ApproxFun.jl) package also performs Chebyshev interpolation and many other tasks.   [BasicInterpolators.jl](https://github.com/markmbaum/BasicInterpolators.jl) also provides Chebyshev interpolation in 1d and 2d, and [Surrogates.jl](https://github.com/SciML/Surrogates.jl) provides some other interpolation schemes.