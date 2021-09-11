"""
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


We also provide a function `chebgradient(c,y)` that returns a tuple `(c(y), ∇c(y))` of
the interpolant and its gradient at a point `y`.

The FastChebInterp package also supports complex and vector-valued functions `f`.  In
this case, `c(y)` returns a vector of interpolants, and one can use `chebjacobian(c,y)`
to compute the tuple `(c(y), J(y))` of the interpolant and its Jacobian matrix at `y`.
"""
module FastChebInterp

export chebpoints, chebinterp, chebinterp_v1, chebjacobian, chebgradient, chebregression

using StaticArrays
import FFTW

"""
    ChebPoly

A multidimensional Chebyshev-polynomial interpolation object.
Given a `c::ChebPoly`, you can evaluate it at a point `x`
with `c(x)`, where `x` is a vector (or a scalar if `c` is 1d).
"""
struct ChebPoly{N,T,Td<:Real}
    coefs::Array{T,N} # chebyshev coefficients
    lb::SVector{N,Td} # lower/upper bounds
    ub::SVector{N,Td} #    of the domain
end

function Base.show(io::IO, c::ChebPoly)
    print(io, "Chebyshev order ", map(i->i-1,size(c.coefs)), " polynomial on ",
          '[', c.lb[1], ',', c.ub[1], ']')
    for i = 2:length(c.lb)
        print(io, " × [", c.lb[i], ',', c.ub[i], ']')
    end
end
Base.ndims(c::ChebPoly) = ndims(c.coefs)

include("interp.jl")
include("regression.jl")
include("eval.jl")

end # module
