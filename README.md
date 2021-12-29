# FastChebInterp

[![Build Status](https://travis-ci.org/stevengj/FastChebInterp.jl.svg?branch=master)](https://travis-ci.org/stevengj/FastChebInterp.jl)

Fast multidimensional Chebyshev interpolation on a hypercube (Cartesian-product)
domain, using a separable (tensor-product) grid of Chebyshev interpolation points, as well as Chebyshev regression (least-square fits) from an arbitrary set of points.   In both cases we support arbitrary dimensionality, complex and vector-valued functions, and fast derivative and Jacobian computation.

## Usage

For domain upper and lower bounds `lb` and `ub`, and a given `order`
tuple, you would create an interpolator for a function `f` via:
```jl
using FastChebInterp
x = chebpoints(order, lb, ub) # an array of `SVector` from [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl), or scalars in 1d
c = chebinterp(f.(x), lb, ub)
```
and then evaluate the interpolant for a point `y` (a vector)
via `c(y)`.

We also provide a function `chebgradient(c,y)` that returns a tuple `(c(y), ∇c(y))` of
the interpolant and its gradient at a point `y`.  (You can also use automatic differentiation, e.g. via the [ForwardDiff.jl package](https://github.com/JuliaDiff/ForwardDiff.jl),
but `chebgradient` is slightly faster and also supports derivatives of complex-valued functions, unlike ForwardDiff.)

The FastChebInterp package also supports complex and vector-valued functions `f`.  In
this case, `c(y)` returns a vector of interpolants, and one can use `chebjacobian(c,y)`
to compute the tuple `(c(y), J(y))` of the interpolant and its Jacobian matrix at `y`.

### Regression from arbitrary points

We also have a function `chebregression(x, y, [lb, ub,], order)` that
can perform multidimensional Chebyshev least-square fitting.  It
returns a Chebyshev polynomial of a given `order` (tuple) fit
to a set of points `x[i]` and values `y[i]`, optionally in a box
with bounds `lb, ub` (which default to bounding box for `x`).

### 1d Example

Here is an example interpolating the (highly oscillatory) 1d function `f(x) = sin(2x + 3cos(4x))` for `0 ≤ x ≤ 10`, with a degree-200 Chebyshev polynomial
```jl
f(x) = sin(2x + 3cos(4x))
x = chebpoints(200, 0, 10)
c = chebinterp(f.(x), 0, 10)
```
We can then compare the "exact" function and its derivative at a set of points:
```jl
julia> xx = 0:0.1:10;

julia> maximum(@. abs(c(xx) - f(xx)))
2.6336643132285342e-5
```
and we see that the interpolant `c` matches `f` to about five decimal digits.

The function `chebgradient` returns both the interpolant and its derivative, e.g. at `x = 1.234`, and we can compare it to the exact values
```jl
julia> chebgradient(c, x)
(0.008334535719968672, -13.700695443638699)

julia> f(x) # exact function value
0.008336024670192028

julia> cos(2x + 3cos(4x)) * (2 - 12sin(4x)) # exact derivative
-13.700760631142602
```

Interpolation is most efficient and accurate if we evaluate our function at the points given by `chebpoints`.   However, we can also perform least-square polynomial fitting (in the Chebyshev basis, which is well behaved even at high degree) from an *arbitrary* set of points — this is useful if the points were specified externally, or if we want to "smooth" the data by fitting to a polynomial of lower degree than for interpolation.    For example, we can fit the same function above, again to a degree-200 Chebyshev polynomial, using 10000 *random* points in the domain:
```jl 
xr = rand(10000) * 10 # 10000 uniform random points in [0, 10]
c = chebregression(xr, f.(xr), 0, 10, 200) # fit to a degree-200 polynomial
```jl
julia> maximum(@. abs(c(xx) - f(xx)))
1.4655330320523241e-5
```

### 2d Example

Here is a 2d example, interpolating the function `g(x) = sin(x₁ + cos(x₂))` for `1 ≤ x₁ ≤ 2` and `3 ≤ x₂ ≤ 4`, using order 10 in the `x₁` direction and order 20 in the `x₂` direction:
```jl
g(x) = sin(x[1] + cos(x[2]))
lb, ub = [1,3], [2, 4] # lower and upper bounds of the domain, respectively
x = chebpoints((10,20), lb, ub)
c = chebinterp(g.(x), lb, ub)
```
Let's evaluate the interpolant at an arbitrary point `(1.2, 3.4)` and compare it to the exact value:
```jl
julia> c([1.2, 3.4]) # polynomial interpolant
0.23109384193446084

julia> g([1.2, 3.4]) # exact value
0.23109384193445792

julia> g([1.2, 3.4]) - c([1.2, 3.4])
-2.914335439641036e-15
```
In this case, because the function is smooth and not very wiggly in the domain, our low-degree polynomial suffices to interpolate to nearly machine (`Float64`) precision.

Note that FastChebInterp uses [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) internally to work with vectors, and e.g. a `Vector` like `[1.2, 3.4]` gets converted to an `SVector` internally.  If you are working with many such coordinate vectors, it is often advisable to [use StaticArrays](https://docs.julialang.org/en/v1/manual/performance-tips/#Consider-StaticArrays.jl-for-small-fixed-size-vector/matrix-operations) yourself, in which case you can pass `SVector` coordinates directly to FastChebInterp functions.

If we inspect the `c` object, we see that it is actually of a lower degree than we requested:
```jl
julia> c
Chebyshev order (10, 16) interpolator on [1,2] × [3,4]
```
What happened is that `chebinterp` computed the order-`(10,20)` polynomial as requested, but found that the order `> 16` terms in the `x₂` direction were all smaller than machine precision, so it discarded them (since lower-degree polynomials are cheaper to work with).    You can control this behavior by passing the `tol` keyword argument to `chebinterp`: passing `tol=0` prevents it from discarding any terms, and a larger `tol` can be used to obtain an even lower-degree polynomial:
```jl
julia> chebinterp(g.(x), lb, ub, tol=0) # prevent terms from being dropped
Chebyshev order (10, 20) interpolator on [1,2] × [3,4]

julia> chebinterp(g.(x), lb, ub, tol=0.01) # request only about 1% accuracy
Chebyshev order (2, 2) interpolator on [1,2] × [3,4]
```

The `chebgradient` function now returns the function and its *gradient* (partial derivatives with respect to `x₁` and `x₂`) vector, which we can compare to the analytical gradient:
```jl
julia> chebgradient(c, [1.2, 3.4])
(0.23109384193446084, [0.9729314653248615, 0.248623978845799])

g(x) = sin(x[1] + cos(x[2]))

julia> [cos(1.2 + cos(3.4)), cos(1.2 + cos(3.4)) * -sin(3.4)] # exact gradient
2-element Vector{Float64}:
 0.9729314653252673
 0.24862397884579854
```
and we see that the derivative matches to high accuracy.

## Related packages

This package was inspired by functionality in [ChebyshevApprox.jl](https://github.com/RJDennis/ChebyshevApprox.jl), but was rewritten in order to get more performance and other features.  The [ApproxFun.jl](https://github.com/JuliaApproximation/ApproxFun.jl) package also performs Chebyshev interpolation and many other tasks.   [BasicInterpolators.jl](https://github.com/markmbaum/BasicInterpolators.jl) also provides Chebyshev interpolation in 1d and 2d, and [Surrogates.jl](https://github.com/SciML/Surrogates.jl) provides some other interpolation schemes.
