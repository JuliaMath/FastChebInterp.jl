# "fitting" (actually just interpolating) Chebyshev polynomials
# to functions evaluated at Chebyshev points.

function chebpoint(i::CartesianIndex{N}, order::NTuple{N,Int}, lb::SVector{N}, ub::SVector{N}) where {N}
    T = typeof(float(one(eltype(lb)) * one(eltype(ub))))
    @. lb + (1 + cos(T($SVector($Tuple(i))) * π / $SVector(ifelse.(iszero.(order),2,order)))) * (ub - lb) * $(T(0.5))
end

chebpoints(order::NTuple{N,Int}, lb::SVector{N}, ub::SVector{N}) where {N} =
    [chebpoint(i,order,lb,ub) for i in CartesianIndices(map(n -> n==0 ? (1:1) : (0:n), order))]

"""
    chebpoints(order, lb, ub)

Return an array of Chebyshev points (as `SVector` values) for
the given `order` (an array or tuple of polynomial degrees),
and hypercube lower-and upper-bound vectors `lb` and `ub`.
If `ub` and `lb` are numbers, returns an array of numbers.

These are the points where you should evaluate a function
in order to create a Chebyshev interpolant with `chebinterp`.

(Note that the number of points along each dimension is `1 +`
the order in that direction.)
"""
function chebpoints(order, lb, ub)
    N = length(order)
    N == length(lb) == length(ub) || throw(DimensionMismatch())
    all(≥(0), order) || throw(ArgumentError("invalid negative order $order"))
    return chebpoints(NTuple{N,Int}(order), SVector{N}(lb), SVector{N}(ub))
end

# return array of scalars in 1d
chebpoints(order::Integer, lb::Real, ub::Real) =
    first.(chebpoints(order, SVector(lb), SVector(ub)))

# O(n log n) method to compute Chebyshev coefficients
function chebcoefs(vals::AbstractArray{<:Number,N}) where {N}
     # type-I DCT, except for size-1 dimensions where we want identity
    kind = map(n -> n > 1 ? FFTW.REDFT00 : FFTW.DHT, size(vals))
    coefs = FFTW.r2r(vals, kind)

    # renormalize the result to obtain the conventional
    # Chebyshev-polnomial coefficients
    s = size(coefs)
    coefs ./= prod(map(n -> n > 1 ? 2(n-1) : 1, s))
    for dim = 1:N
        if size(coefs, dim) > 1
            coefs[CartesianIndices(ntuple(i -> i == dim ? (2:s[i]-1) : (1:s[i]), Val{N}()))] .*= 2
        end
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
            r = let n=n; ntuple(i -> i == dim ? (n:n) : (1:size(coefs,i)), Val{N}()); end
            any(c -> infnorm(c) ≥ abstol, @view coefs[CartesianIndices(r)]) && break
            n -= 1
        end
        n
    end
    return coefs[CartesianIndices(map(n -> 1:n, newsize))]
end

function chebinterp(vals::AbstractArray{<:Any,N}, lb::SVector{N}, ub::SVector{N};
                    tol::Real=epsvals(vals), extrapolate::Bool=false) where {N}
    Td = promote_type(eltype(lb), eltype(ub))
    coefs = droptol(chebcoefs(vals), tol)
    return ChebPoly{N,eltype(coefs),Td}(coefs, SVector{N,Td}(lb), SVector{N,Td}(ub), extrapolate)
end

# precision for float(vals), handling arrays of numbers and arrays of arrays of numbers
epsvals(vals) = eps(float(real(eltype(eltype(vals)))))

"""
    chebinterp(vals, lb, ub; tol=eps, extrapolate=false)

Given a multidimensional array `vals` of function values (at
points corresponding to the coordinates returned by `chebpoints`),
and arrays `lb` and `ub` of the lower and upper coordinate bounds
of the domain in each direction, returns a Chebyshev interpolation
object (a `ChebPoly`).

This object `c = chebinterp(vals, lb, ub)` can be used to
evaluate the interpolating polynomial at a point `x` via
`c(x)`.

The elements of `vals` can be vectors as well as numbers, in order
to interpolate vector-valued functions (i.e. to interpolate several
functions at once).

The `tol` argument specifies a relative tolerance below which
Chebyshev coefficients are dropped; it defaults to machine precision
for the precision of `float(vals)`.   Passing `tol=0` will keep
all coefficients up to the order passed to `chebpoints`.

If `extrapolate=true` is passed, then the resulting Chebyshev polynomial
can be evaluated outside the `[lb,ub]` domain; otherwise, an error is
thrown for out-of-domain points.
"""
chebinterp(vals::AbstractArray{<:Any,N}, lb, ub;
    tol::Real=epsvals(vals), extrapolate::Bool=false) where {N} =
    chebinterp(vals, SVector{N}(lb), SVector{N}(ub);
               tol=tol, extrapolate=extrapolate)

"""
    chebinterp_v1(vals, lb, ub; tol=eps, extrapolate=false)

Like `chebinterp(vals, lb, ub)`, but slices off the *first* dimension of `vals`
and treats it as a vector of values to interpolate.

For example, if `vals` is a 2×31×32 array of numbers, then it is treated
equivalently to calling `chebinterp` with a 31×32 array of 2-component vectors.

(This function is mainly useful for calling from Python, where arrays
of vectors are painful to construct.)
"""
chebinterp_v1(vals::AbstractArray{T}, lb, ub;
              tol::Real=epsvals(vals), extrapolate::Bool=false) where {T<:Number} =
    chebinterp(dropdims(reinterpret(SVector{size(vals,1),T}, Array(vals)), dims=1),
               lb, ub; tol=tol, extrapolate=extrapolate)
