# Computing roots of Chebyshev real-valued polynomials in 1d via colleague matrices on
# recursively subdivided intervals, as described in Trefethen, "Approximation
# Theory and Approximation Practice", chapter 18.  https://epubs.siam.org/doi/10.1137/1.9781611975949
#
# Some additional implementation tricks were inspired by the chebfun source
# code: https://github.com/chebfun/chebfun/blob/7574c77680d7e82b79626300bf255498271a72df/%40chebtech/roots.m

using LinearAlgebra
export colleague_matrix

# colleague matrix for 1d array of coefficients, assuming
# length > 2 and trailing zero coefficients have already been dropped.
function _colleague_matrix(coefs::AbstractVector{<:Real})
    n = length(coefs)
    n > 2 || throw(ArgumentError("length(coefs) = $(length(coefs)) must be > 2"))
    iszero(coefs[end]) && throw(ArgumentError("trailing coefficient must be nonzero"))

    # colleage matrix, transposed to make it upper-Hessenberg (which doesn't change eigenvalues)
    halves = fill(one(eltype(coefs))/2, n-2)
    C = diagm(1 => halves, -1 => halves)
    C[2,1] = 1
    @views C[:,end] .-= coefs[1:end-1] ./ 2coefs[end]
    return C
end

"""
    colleague_matrix(c::ChebPoly{1,<:Real}; tol=5eps)

Return the "colleague matrix" whose eigenvalues are the roots of the 1d real-valued
Chebyshev polynomial `c`, dropping trailing coefficients whose relative contributions
are `< tol` (defaulting to 5 × floating-point `eps`).   Throws an error if the resulting
polynomial degree is less than 2.
"""
function colleague_matrix(c::ChebPoly{1,<:Real}; tol::Real=5*epsvals(c.coefs))
    abstol = sum(infnorm, c.coefs) * tol # absolute tolerance = L1 norm * tol
    n = something(findlast(c -> infnorm(c) ≥ abstol, c.coefs), 1)
    C = _colleague_matrix(@view c.coefs[1:n])
    # scale and shift from [-1,1] to [lb,ub] via C * (ub-lb)/2 + (ub+lb)/2 * I
    C .*= (c.ub[1] - c.lb[1])/2
    diagview(C) .+= (c.ub[1] + c.lb[1])/2
    return C
end
