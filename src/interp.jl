# "fitting" (actually just interpolating) Chebyshev polynomials
# to functions evaluated at Chebyshev points.

function chebpoint(i::CartesianIndex{N}, order::NTuple{N,Int}, lb::SVector{N}, ub::SVector{N}) where {N}
    T = typeof(float(one(eltype(lb)) * one(eltype(ub))))
    @. lb + (1 + cos(T($SVector($Tuple(i))) * π / $SVector(ifelse.(iszero.(order),2,order)))) * (ub - lb) * $(T(0.5))
end

function chebpoints(order::NTuple{N,Int}, lb::SVector{N}, ub::SVector{N}) where {N}
    all(≥(0), order) || throw(ArgumentError("invalid negative order $order"))
    return [chebpoint(i,order,lb,ub) for i in CartesianIndices(map(n -> n==0 ? (1:1) : (0:n), order))]
end

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
    return chebpoints(NTuple{N,Int}(order), SVector{N}(lb), SVector{N}(ub))
end

# return array of scalars in 1d
chebpoints(order::Integer, lb::Real, ub::Real) =
    first.(chebpoints((Int(order),), SVector(lb), SVector(ub)))

# O(n log n) method to compute Chebyshev coefficients
function chebcoefs(vals::AbstractArray{<:Number,N}) where {N}
    all(isfinite, vals) || throw(DomainError("non-finite interpolant value"))

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
    abstol = maximum(infnorm, coefs) * tol # absolute tolerance = Linf norm * tol
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

function chebinterp(vals::AbstractArray{<:Any,N}, lb::SVector{N}, ub::SVector{N}; tol::Real=epsvals(vals)) where {N}
    Td = promote_type(eltype(lb), eltype(ub))
    coefs = droptol(chebcoefs(vals), tol)
    return ChebPoly{N,eltype(coefs),Td}(coefs, SVector{N,Td}(lb), SVector{N,Td}(ub))
end

# precision for float(vals), handling arrays of numbers and arrays of arrays of numbers
epsvals(vals) = eps(float(real(eltype(eltype(vals)))))
epsbounds(lb, ub) = eps(float(real(promote_type(eltype(lb), eltype(ub)))))

"""
    chebinterp(vals, lb, ub; tol=eps)
    chebinterp(f::Function, order, lb, ub; tol=eps)
    chebinterp(f::Function, lb, ub; tol=eps, min_order=(8,..), max_order=(typemax(Int),...))

Given a multidimensional array `vals` of function values (at
points corresponding to the coordinates returned by `chebpoints`),
and arrays `lb` and `ub` of the lower and upper coordinate bounds
of the domain in each direction, returns a Chebyshev interpolation
object (a `ChebPoly`).

Alternatively, one can supply a function `f` and an `order` (an integer
or a tuple of integers), and it will call [`chebpoints`](@ref) for you
to obtain the Chebyshev points and then compute `vals` by evaluating
`f` at these points.

If a function `f` is supplied and the `order` argument is omitted, it will adaptively
determine the order by repeatedly doubling it until `tol` is achieved
or `max_order` is reached, starting at `min_order` (which defaults to `8` or
a tuple of `8`s in each dimension; this might need to be increased for
highly oscillatory functions).   This feature is best used for smooth functions.

This object `c = chebinterp(...)` can be used to
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
chebinterp(vals::AbstractArray{<:Any,N}, lb, ub; tol::Real=epsvals(vals)) where {N} =
    chebinterp(vals, SVector{N}(lb), SVector{N}(ub); tol=tol)

"""
    chebinterp_v1(vals, lb, ub; tol=eps)

Like `chebinterp(vals, lb, ub)`, but slices off the *first* dimension of `vals`
and treats it as a vector of values to interpolate.

For example, if `vals` is a 2×31×32 array of numbers, then it is treated
equivalently to calling `chebinterp` with a 31×32 array of 2-component vectors.

(This function is mainly useful for calling from Python, where arrays
of vectors are painful to construct.)
"""
chebinterp_v1(vals::AbstractArray{T}, lb, ub; tol::Real=epsvals(vals)) where {T<:Number} =
    chebinterp(dropdims(reinterpret(SVector{size(vals,1),T}, Array(vals)), dims=1), lb, ub; tol=tol)

#####################################################################################################
# supplying the function rather than the values

# supplying both function and order:

chebinterp(f::Function, order::NTuple{N,Int}, lb::SVector{N,<:Real}, ub::SVector{N,<:Real}; tol::Real=epsbounds(lb,ub)) where {N} =
    chebinterp(map(f, chebpoints(order, lb, ub)), lb, ub; tol=tol)

chebinterp(f::Function, order::NTuple{N,Int}, lb::AbstractArray{<:Real}, ub::AbstractArray{<:Real}; tol::Real=epsbounds(lb,ub)) where {N} =
    chebinterp(f, order, SVector{N}(lb), SVector{N}(ub); tol=tol)

chebinterp(f::Function, order::Integer, lb::Real, ub::Real; tol::Real=epsbounds(lb,ub)) =
    chebinterp(x -> f(@inbounds x[1]), (Int(order),), SVector(lb), SVector(ub); tol=tol)

## adaptively determine the order by repeated doublying, ala chebfun or approxfun:

function chebinterp(f::Function, lb::SVector{N,<:Real}, ub::SVector{N,<:Real};
                    tol::Real=5epsbounds(lb,ub),
                    min_order::NTuple{N,Int}=ntuple(i->8,Val{N}()),
                    max_order::NTuple{N,Int}=ntuple(i->typemax(Int),Val{N}())) where {N}
    tol > 0 || throw(ArgumentError("tolerance $tol must be > 0"))
    all(min_order .> 0) || throw(ArgumentError("minimum order $min_order must be > 0"))
    all(max_order .>= 0) || throw(ArgumentError("maximum order $max_order must be ≥ 0"))
    order = min.(min_order, max_order)
    while true
        # in principle we could re-use function evaluations when doubling the order,
        # but that would greatly complicate the code and only saves a factor of 2
        c = chebinterp(map(f, chebpoints(order, lb, ub)), lb, ub; tol=tol)
        order_done = (size(c.coefs) .- 1 .< order) .| (order .== max_order)
        all(order_done) && return c
        order = ifelse.(order_done, order, min.(max_order, nextpow.(2, order .* 2)))
    end
end

function chebinterp(f::Function, lb::AbstractArray{<:Real}, ub::AbstractArray{<:Real};
                    tol::Real=5epsbounds(lb,ub),
                    min_order=fill(8, length(lb)), max_order=fill(typemax(Int), length(lb)))
    N = length(lb)
    N == length(ub) == length(min_order) == length(max_order) || throw(DimensionMismatch("dimensions must all == $N"))
    Base.require_one_based_indexing(min_order, max_order)
    chebinterp(f, SVector{N}(lb), SVector{N}(ub);
               tol=tol, min_order=ntuple(i -> Int(min_order[i]), N), max_order=ntuple(i -> Int(max_order[i]), N))
end

chebinterp(f::Function, lb::Real, ub::Real; tol::Real=5epsbounds(lb,ub), min_order::Integer=8, max_order::Integer=typemax(Int)) =
    chebinterp(x -> f(@inbounds x[1]), SVector(lb), SVector(ub); tol=tol, min_order=(Int(min_order),), max_order=(Int(max_order),))
