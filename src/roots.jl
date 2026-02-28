# Computing roots of Chebyshev real-valued polynomials in 1d via colleague matrices on
# recursively subdivided intervals, as described in Trefethen, "Approximation
# Theory and Approximation Practice", chapter 18.  https://epubs.siam.org/doi/10.1137/1.9781611975949
#
# Some additional implementation tricks were inspired by the chebfun source
# code: https://github.com/chebfun/chebfun/blob/7574c77680d7e82b79626300bf255498271a72df/%40chebtech/roots.m

using LinearAlgebra
export colleague_matrix, roots

# colleague matrix for 1d array of Chebyshev coefficients, assuming
# trailing zero coefficients have already been dropped.
function _colleague_matrix(coefs::AbstractVector{<:Number})
    n = length(coefs)
    n <= 1 && return Matrix{float(eltype(coefs))}(undef,0,0) # 0×0 case (no roots)
    iszero(coefs[end]) && throw(ArgumentError("trailing coefficient must be nonzero"))

    if n == 2 # trivial 1×1 (degree 1) case
        C = Matrix{float(eltype(coefs))}(undef,1,1)
        C[1,1] = float(-coefs[1])/coefs[2]
        return C
    else
        # colleague matrix, transposed to make it upper-Hessenberg (which doesn't change eigenvalues)
        halves = fill(one(float(eltype(coefs)))/2, n-2)
        C = diagm(1 => halves, -1 => halves)
        C[2,1] = 1
        @views C[:,end] .-= coefs[1:end-1] ./ 2coefs[end]
        return C
    end
end


if !isdefined(LinearAlgebra, :diagview) # added in Julia 1.12
    if !isdefined(LinearAlgebra, :diagind)
        diagind(A::Matrix) = range(1, step=size(A,1)+1, length=min(size(A)...))
    end
    diagview(A::Matrix) = @view A[diagind(A)]
end

function _colleague_matrix(coefs::AbstractVector{<:Number}, lb::Real, ub::Real)
    C = _colleague_matrix(coefs)
    # scale and shift from [-1,1] to [lb,ub] via C * (ub-lb)/2 + (ub+lb)/2 * I
    C .*= (ub - lb)/2
    diagview(C) .+= (ub + lb)/2
    return C
end

"""
    colleague_matrix(c::ChebPoly{1,<:Number}; tol=5eps)

Return the "colleague matrix" whose eigenvalues are the roots of the 1d real-valued
Chebyshev polynomial `c`, dropping trailing coefficients whose relative contributions
are `< tol` (defaulting to 5 × floating-point `eps`).
"""
function colleague_matrix(c::ChebPoly{1,<:Number}; tol::Real=5*epsvals(c.coefs))
    abstol = sum(abs, c.coefs) * tol # absolute tolerance = L1 norm * tol
    n = something(findlast(c -> abs(c) ≥ abstol, c.coefs), 1)
    return UpperHessenberg(_colleague_matrix(@view(c.coefs[1:n]), c.lb[1], c.ub[1]))
end

function filter_roots(c::ChebPoly{1,<:Real}, roots::AbstractVector{<:Number})
    @inbounds lb, ub = c.lb[1], c.ub[1]
    htol = eps(float(ub - lb)) * 100 # similar to chebfun
    return [ clamp(real(r), lb, ub) for r in roots if abs(imag(r)) < htol && lb - htol <= real(r) <= ub + htol ]
end

if isdefined(LinearAlgebra.LAPACK, :hseqr!) # added in Julia 1.10
    # see also LinearAlgebra.jl#1557 - optimized in-place eigenvalues for upper-Hessenberg colleague matrix
    function hesseneigvals!(C::UpperHessenberg{T,Matrix{T}}) where {T<:Union{LinearAlgebra.BlasReal, LinearAlgebra.BlasComplex}}
        ilo, ihi, _ = LinearAlgebra.LAPACK.gebal!('S', triu!(C.data, -1))
        return sort!(LinearAlgebra.LAPACK.hseqr!('E', 'N', 1, size(C,1), C.data, C.data)[3], by=reim)
    end
end
hesseneigvals!(C::UpperHessenberg{T,Matrix{T}}) where {T} = eigvals!(triu!(C.data, -1))

"""
    roots(c::ChebPoly{1,<:Real}; tol=5eps, maxsize::Integer=50)

Returns a sorted array of the real roots of `c` on the interval `[lb,ub]` (the lower and
upper bounds of the interpolation domain).

Uses a colleague-matrix method combined with recursive subdivision of the domain
to keep the maximum matrix size `≤ maxsize`, following an algorithm described by
Trefethen (*Approximation Theory and Approximation Practice*, ch. 18).  The `tol`
argument is a relative tolerance for dropping small polynomial coefficients, defaulting
to `5eps` where `eps` is the precision of `c`.
"""
function roots(c::ChebPoly{1,<:Real}; tol::Real=5*epsvals(c.coefs), maxsize::Integer=50)
    tol > 0 || throw(ArgumentError("tolerance $tol for truncating coefficients must be > 0"))
    maxsize > 0 || throw(ArgumentError("maxsize $maxsize must be > 0"))
    abstol = maximum(abs, c.coefs) * tol # absolute tolerance = Linf norm * tol
    n = something(findlast(c -> abs(c) ≥ abstol, c.coefs), 1)
    if n <= maxsize
        λ = hesseneigvals!(UpperHessenberg(_colleague_matrix(@view c.coefs[1:n])))
        @inbounds λ .= (λ .+ 1) .* ((c.ub[1] - c.lb[1])/2) .+ c.lb[1] # scale and shift to [lb,ub]
        return filter_roots(c, λ)
    else
        # roughly halve the domain, constructing new Chebyshev polynomials on each half, and
        # call roots recursively.  Following chebfun, we split at an arbitrary point 0.004849834917525
        # on [-1,1] rather than at 0 to avoid introducing additional spurious roots (since 0 is
        # often a special point by symmetry).
        @inbounds split = oftype(float(c.lb[1]), 1.004849834917525) * ((c.ub[1] - c.lb[1])/2) + c.lb[1]

        # pick a fast order for the recursive DCT, should be highly composite, say 2^m
        order = nextpow(2, n-1)
        c1 = chebinterp(c, order, c.lb[1], split; tol=2tol)
        c2 = chebinterp(c, order, split, c.ub[1]; tol=2tol)
        return vcat(roots(c1; tol=2tol, maxsize=maxsize), roots(c2; tol=2tol, maxsize=maxsize))
    end
end
