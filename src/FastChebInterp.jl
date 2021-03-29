"""
Fast multidimensional Chebyshev interpolation on a hypercube (Cartesian-product)
domain, using a separable (tensor-product) grid of Chebyshev interpolation points.

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
"""
module FastChebInterp

export chebpoints, chebfit, chebfitv1, chebjacobian, chebgradient

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
    print(io, "Chebyshev order ", map(i->i-1,size(c.coefs)), " interpolator on ",
          '[', c.lb[1], ',', c.ub[1], ']')
    for i = 2:length(c.lb)
        print(io, " × [", c.lb[i], ',', c.ub[i], ']')
    end
end
Base.ndims(c::ChebPoly) = ndims(c.coefs)

chebpoint(i::CartesianIndex{N}, order::NTuple{N,Int}, lb::SVector{N}, ub::SVector{N}) where {N} =
    @. lb + (1 + cos($SVector($Tuple(i)) * π / $SVector(order))) * (ub - lb) * 0.5

chebpoints(order::NTuple{N,Int}, lb::SVector{N}, ub::SVector{N}) where {N} =
    [chebpoint(i,order,lb,ub) for i in CartesianIndices(map(n -> 0:n, order))]

"""
    chebpoints(order, lb, ub)

Return an array of Chebyshev points (as `SVector` values) for
the given `order` (an array or tuple of polynomial degrees),
and hypercube lower-and upper-bound vectors `lb` and `ub`.
If `ub` and `lb` are numbers, returns an array of numbers.

These are the points where you should evaluate a function
in order to create a Chebyshev interpolant with `chebfit`.

(Note that the number of points along each dimension is `1 +`
the order in that direction.)
"""
function chebpoints(order, lb, ub)
    N = length(order)
    N == length(lb) == length(ub) || throw(DimensionMismatch())
    return chebpoints(NTuple{N,Int}(order), SVector{N}(lb), SVector{N}(ub))
end

# return array of scalars in 1d
chebpoints(order::Integer, lb::Real, ub::Real) =
    first.(chebpoints(order, SVector(lb), SVector(ub)))

# O(n log n) method to compute Chebyshev coefficients
function chebcoefs(vals::AbstractArray{<:Number,N}) where {N}
    coefs = FFTW.r2r(vals, FFTW.REDFT00) # type-I DCT

    # renormalize the result to obtain the conventional
    # Chebyshev-polnomial coefficients
    s = size(coefs)
    coefs ./= prod(map(n -> 2(n-1), s))
    for dim = 1:N
        coefs[CartesianIndices(ntuple(i -> i == dim ? (2:s[i]-1) : (1:s[i]), Val{N}()))] .*= 2
    end

    return coefs
end

function chebcoefs(vals::AbstractArray{<:SVector{K}}) where {K}
    # TODO: in principle we could call FFTW's low-level interface
    # to perform all the transforms simultaneously, rather than
    # transforming each component separately.
    coefs = ntuple(i -> chebcoefs([v[i] for v in vals]), Val{K}())
    return SVector{K}.(coefs...)
end

function chebcoefs(vals::AbstractArray{<:AbstractVector})
    K, K′ = extrema(length, vals)
    K == K′ || throw(ArgumentError("array elements must all be of the same length"))
    return chebcoefs(SVector{K}.(vals))
end

# norm for tolerance tests of chebyshev coefficients
infnorm(x::Number) = abs(x)
infnorm(x::AbstractArray) = maximum(abs, x)

function droptol(coefs::Array{<:Any,N}, tol::Real) where {N}
    abstol = maximum(infnorm, coefs) * tol # absolute tolerance
    all(c -> infnorm(c) ≥ abstol, coefs) && return coefs # nothing to drop

    # compute the new size along each dimension by checking
    # the maximum index that cannot be dropped along each dim
    newsize = ntuple(Val{N}()) do dim
        n = size(coefs, dim)
        while n > 1
            r = ntuple(i -> i == dim ? (n:n) : (1:size(coefs,i)), Val{N}())
            any(c -> infnorm(c) ≥ abstol, @view coefs[CartesianIndices(r)]) && break
            n -= 1
        end
        n
    end
    return coefs[CartesianIndices(map(n -> 1:n, newsize))]
end

function chebfit(vals::AbstractArray{<:Any,N}, lb::SVector{N}, ub::SVector{N}; tol::Real=epsvals(vals)) where {N}
    Td = promote_type(eltype(lb), eltype(ub))
    coefs = droptol(chebcoefs(vals), tol)
    return ChebPoly{N,eltype(coefs),Td}(coefs, SVector{N,Td}(lb), SVector{N,Td}(ub))
end

# precision for float(vals), handling arrays of numbers and arrays of arrays of numbers
epsvals(vals) = eps(float(real(eltype(eltype(vals)))))

"""
    chebfit(vals, lb, ub; tol=eps)

Given a multidimensional array `vals` of function values (at
points corresponding to the coordinates returned by `chebpoints`),
and arrays `lb` and `ub` of the lower and upper coordinate bounds
of the domain in each direction, returns a Chebyshev interpolation
object (a `ChebPoly`).

This object `c = chebfit(vals, lb, ub)` can be used to
evaluate the interpolating polynomial at a point `x` via
`c(x)`.

The elements of `vals` can be vectors as well as numbers, in order
to interpolate vector-valued functions (i.e. to interpolate several
functions at once).

The `tol` argument specifies a relative tolerance below which
Chebyshev coefficients are dropped; it defaults to machine precision
for the precision of `float(vals)`.   Passing `tol=0` will keep
all coefficients up to the order passed to `chebpoints`.
"""
chebfit(vals::AbstractArray{<:Any,N}, lb, ub; tol::Real=epsvals(vals)) where {N} =
    chebfit(vals, SVector{N}(lb), SVector{N}(ub); tol=tol)

"""
    chebfitv1(vals, lb, ub; tol=eps)

Like `chebfit(vals, lb, ub)`, but slices off the *first* dimension of `vals`
and treats it as a vector of values to interpolate.

For example, if `vals` is a 2×31×32 array of numbers, then it is treated
equivalently to calling `chebfit` with a 31×32 array of 2-component vectors.

(This function is mainly useful for calling from Python, where arrays
of vectors are painful to construct.)
"""
chebfitv1(vals::AbstractArray{T}, lb, ub; tol::Real=epsvals(vals)) where {T<:Number} =
    chebfit(dropdims(reinterpret(SVector{size(vals,1),T}, Array(vals)), dims=1), lb, ub; tol=tol)

"""
    interpolate(x, c::Array{T,N}, ::Val{dim}, i1, len)

Low-level interpolation function, which performs a
multidimensional Clenshaw recurrence by recursing on
the coefficient (`c`) array dimension `dim` (from `N` to `1`).   The
current dimension (via column-major order) is accessed
by `c[i1 + (i-1)*Δi]`, i.e. `i1` is the starting index and
`Δi = len ÷ size(c,dim)` is the stride.   `len` is the
product of `size(c)[1:dim]`. The interpolation point `x`
should lie within [-1,+1] in each coordinate.
"""
function interpolate(x::SVector{N}, c::Array{<:Any,N}, ::Val{dim}, i1, len) where {N,dim}
    n = size(c,dim)
    @inbounds xd = x[dim]
    if dim == 1
        c₁ = c[i1]
        if n ≤ 2
            n == 1 && return c₁ + one(xd) * zero(c₁)
            return c₁ + xd*c[i1]
        end
        @inbounds bₖ = c[i1+(n-2)] + 2xd*c[i1+(n-1)]
        @inbounds bₖ₊₁ = oftype(bₖ, c[i1+(n-1)])
        for j = n-3:-1:1
            @inbounds bⱼ = c[i1+j] + 2xd*bₖ - bₖ₊₁
            bₖ, bₖ₊₁ = bⱼ, bₖ
        end
        return c₁ + xd*bₖ - bₖ₊₁
    else
        Δi = len ÷ n # column-major stride of current dimension

        # we recurse downward on dim for cache locality,
        # since earlier dimensions are contiguous
        dim′ = Val{dim-1}()

        c₁ = interpolate(x, c, dim′, i1, Δi)
        if n ≤ 2
            n == 1 && return c₁ + one(xd) * zero(c₁)
            c₂ = interpolate(x, c, dim′, i1+Δi, Δi)
            return c₁ + xd*c₂
        end
        cₙ₋₁ = interpolate(x, c, dim′, i1+(n-2)*Δi, Δi)
        cₙ = interpolate(x, c, dim′, i1+(n-1)*Δi, Δi)
        bₖ = cₙ₋₁ + 2xd*cₙ
        bₖ₊₁ = oftype(bₖ, cₙ)
        for j = n-3:-1:1
            cⱼ = interpolate(x, c, dim′, i1+j*Δi, Δi)
            bⱼ = cⱼ + 2xd*bₖ - bₖ₊₁
            bₖ, bₖ₊₁ = bⱼ, bₖ
        end
        return c₁ + xd*bₖ - bₖ₊₁
    end
end

"""
    (interp::ChebPoly)(x)

Evaluate the Chebyshev polynomial given by `interp` at the point `x`.
"""
function (interp::ChebPoly{N})(x::SVector{N,<:Real}) where {N}
    x0 = @. (x - interp.lb) * 2 / (interp.ub - interp.lb) - 1
    all(abs.(x0) .≤ 1) || throw(ArgumentError("$x not in domain"))
    return interpolate(x0, interp.coefs, Val{N}(), 1, length(interp.coefs))
end

(interp::ChebPoly{N})(x::AbstractVector{<:Real}) where {N} = interp(SVector{N}(x))
(interp::ChebPoly{1})(x::Real) = interp(SVector{1}(x))


"""
    Jinterpolate(x, c::Array{T,N}, ::Val{dim}, i1, len)

Similar to `interpolate` above, but returns a tuple `(v,J)` of the
interpolated value `v` and the Jacobian `J` with respect to `x[1:dim]`.
"""
function Jinterpolate(x::SVector{N}, c::Array{<:Any,N}, ::Val{dim}, i1, len) where {N,dim}
    n = size(c,dim)
    @inbounds xd = x[dim]
    if dim == 1
        c₁ = c[i1]
        if n ≤ 2
            n == 1 && return c₁ + one(xd) * zero(c₁), hcat(SVector(zero(c₁) / oneunit(xd)))
            return c₁ + xd*c[i1], hcat(SVector(c[i1]))
        end
        @inbounds cₙ₋₁ = c[i1+(n-2)]
        @inbounds cₙ = c[i1+(n-1)]
        bₖ = cₙ₋₁ + xd*(bₖ′ = 2cₙ)
        bₖ₊₁ = oftype(bₖ, cₙ)
        bₖ₊₁′ = zero(bₖ₊₁)
        for j = n-3:-1:1
            @inbounds cⱼ = c[i1+j]
            bⱼ = cⱼ + xd*(2bₖ) - bₖ₊₁
            bⱼ′ = xd*(2bₖ′) + (2bₖ) - bₖ₊₁′
            bₖ, bₖ₊₁ = bⱼ, bₖ
            bₖ′, bₖ₊₁′ = bⱼ′, bₖ′
        end
        return c₁ + xd*bₖ - bₖ₊₁, hcat(SVector(bₖ + xd*bₖ′ - bₖ₊₁′))
    else
        Δi = len ÷ n # column-major stride of current dimension

        # we recurse downward on dim for cache locality,
        # since earlier dimensions are contiguous
        dim′ = Val{dim-1}()

        c₁,Jc₁ = Jinterpolate(x, c, dim′, i1, Δi)
        if n ≤ 2
            n == 1 && return c₁ + one(xd) * zero(c₁), hcat(Jc₁, SVector(zero(c₁) / oneunit(xd)))
            c₂,Jc₂ = Jinterpolate(x, c, dim′, i1+Δi, Δi)
            return c₁ + xd*c₂, hcat(Jc₁ + xd*Jc₂, SVector(c[i1]))
        end
        cₙ₋₁,Jcₙ₋₁ = Jinterpolate(x, c, dim′, i1+(n-2)*Δi, Δi)
        cₙ,Jcₙ = Jinterpolate(x, c, dim′, i1+(n-1)*Δi, Δi)
        bₖ = cₙ₋₁ + xd*(bₖ′ = 2cₙ)
        Jbₖ = Jcₙ₋₁ + 2xd*Jcₙ
        bₖ₊₁ = oftype(bₖ, cₙ)
        bₖ₊₁′ = zero(bₖ₊₁)
        Jbₖ₊₁ = oftype(Jbₖ, Jcₙ)
        for j = n-3:-1:1
            cⱼ,Jcⱼ = Jinterpolate(x, c, dim′, i1+j*Δi, Δi)
            bⱼ = cⱼ + xd*(2bₖ) - bₖ₊₁
            bⱼ′ = xd*(2bₖ′) + (2bₖ) - bₖ₊₁′
            Jbⱼ = Jcⱼ + xd*(2Jbₖ) - Jbₖ₊₁
            bₖ, bₖ₊₁ = bⱼ, bₖ
            bₖ′, bₖ₊₁′ = bⱼ′, bₖ′
            Jbₖ, Jbₖ₊₁ = Jbⱼ, Jbₖ
        end
        return c₁ + xd*bₖ - bₖ₊₁, hcat(Jc₁ + xd*Jbₖ - Jbₖ₊₁, SVector(bₖ + xd*bₖ′ - bₖ₊₁′))
    end
end


"""
    chebjacobian(c::ChebPoly, x)

Return a tuple `(v, J)` where `v` is the value `c(x)` of the Chebyshev
polynomial `c` at `x`, and `J` is the Jacobian of this value with respect to `x`.

That is, if `v` is a vector, then `J` is a matrix of `length(v)` × `length(x)`
giving the derivatives of each component.  If `v` is a scalar, then `J`
is a 1-row matrix; in this case you may wish to call `chebgradient` instead.
"""
function chebjacobian(c::ChebPoly{N}, x::SVector{N,<:Real}) where {N}
    x0 = @. (x - c.lb) * 2 / (c.ub - c.lb) - 1
    all(abs.(x0) .≤ 1) || throw(ArgumentError("$x not in domain"))
    v, J = Jinterpolate(x0, c.coefs, Val{N}(), 1, length(c.coefs))
    return v, J .* 2 ./ (c.ub .- c.lb)'
end

chebjacobian(c::ChebPoly{N}, x::AbstractVector{<:Real}) where {N} = chebjacobian(c, SVector{N}(x))
chebjacobian(c::ChebPoly{1}, x::Real) = chebjacobian(c, SVector{1}(x))

"""
    chebgradient(c::ChebPoly, x)

Return a tuple `(v, ∇v)` where `v` is the value `c(x)` of the Chebyshev
polynomial `c` at `x`, and `∇v` is the gradient of this value with respect to `x`.

(Requires `c` to be a scalar-valued polynomial; if `c` is vector-valued, you
should use `chebjacobian` instead.)

If `x` is a scalar, returns a scalar `∇v`, i.e. `(v, ∂v/∂x).
"""
function chebgradient(c::ChebPoly, x)
    v, J = chebjacobian(c, x)
    length(v) == 1 || throw(DimensionMismatch())
    return v, vec(J)
end

function chebgradient(c::ChebPoly{1}, x::Real)
    v, ∇v = chebgradient(c, SVector{1}(x))
    return v, ∇v[1]
end

end # module
