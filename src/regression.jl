# Chebyshev regression: least-square fits of data
# to multidimensional Chebyshev polynomials.

# assemble Chebyshev-Vandermonde matrix
function _chebvandermonde(x::AbstractVector{SVector{N,Td}}, lb::SVector{N,Td}, ub::SVector{N,Td}, order::NTuple{N,Int}) where {N,Td<:Real}
    # (TODO: this algorithm is O(length(x) * length(c)²),
    #  but it should be possible to do it in linear time.
    #  However, the A \ Y step is also O(mn²), so this
    #  only affects the constant factor in the complexity.)
    c = ChebPoly{N,Td,Td}(zeros(Td, order .+ 1), lb, ub)
    A = Array{Td}(undef, length(x), length(c.coefs))
    for i = 1:length(c.coefs)
        c.coefs[i] = 1 # basis function
        for j = 1:length(x)
            A[j,i] = c(x[j + (firstindex(x)-1)])
        end
        c.coefs[i] = 0 # reset
    end
    return A
end

# wrapper around _chebvandermonde for testing convenience
chebvandermonde(x::AbstractVector{SVector{N,Td}}, lb::SVector{N,Td}, ub::SVector{N,Td}, order::NTuple{N,Int}) where {N,Td<:Real} =
    _chebvandermonde(x, lb, ub, order)

# optimized method for 1d case
function chebvandermonde(x::AbstractVector{SVector{1,Td}}, lb::SVector{1,Td}, ub::SVector{1,Td}, order::NTuple{1,Int}) where {Td<:Real}
    lb1, ub1, o1 = lb[1], ub[1], order[1]
    o1 >= 0 || throw(ArgumentError("order $o1 must be nonnegative"))
    A = Array{Td}(undef, length(x), o1+1)
    for j = 1:length(x)
        xⱼ = (x[j][1] - lb1) * 2 / (ub1 - lb1) - 1
        -1 ≤ xⱼ ≤ 1 || throw(ArgumentError("$(x[j][1]) not in domain [$lb1,$ub1]"))
        A[j,1] = Tᵢ₋₂ = 1
        if o1 > 0
            Tᵢ₋₁ = xⱼ
            A[j,2] = Tᵢ₋₁
            twoxⱼ = 2xⱼ
            for i = 3:o1+1 # Chebyshev recurrence
                A[j,i] = Tᵢ = twoxⱼ * Tᵢ₋₁ - Tᵢ₋₂
                Tᵢ₋₂, Tᵢ₋₁ = Tᵢ₋₁, Tᵢ
            end
        end
    end
    return A
end

# convenient API for 1d case
chebvandermonde(x::AbstractVector{Td}, lb::Real, ub::Real, order::Integer) where {Td<:Real} =
    return chebvandermonde(reinterpret(SVector{1,Td}, x), SVector{1,Td}(lb), SVector{1,Td}(ub), (Int(order),))

function chebregression(x::AbstractVector{SVector{N,Td}}, y::AbstractVector{T},
                        lb::SVector{N,Td}, ub::SVector{N,Td}, order::NTuple{N,Int}) where {N,Td<:Real,T<:Union{StaticArray,Number}}
    length(x) == length(y) || throw(DimensionMismatch())
    length(x) ≥ prod(order .+ 1) || throw(ArgumentError("not enough data points $(length(x)) to fit to order $order"))

    # assemble rhs as matrix
    Y = transpose(reshape(reinterpret(eltype(T), y), :, length(y)))

    # assemble lhs matrix
    A = chebvandermonde(x, lb, ub, order)

    # least-square solution
    C = A \ Y

    # rearrange C into a ChebPoly
    Tc = typeof(zero(T) * one(eltype(Y)))
    coefs = Array{Tc,N}(reshape(reinterpret(Tc, vec(transpose(C))), order .+ 1))
    return ChebPoly{N,Tc,Td}(coefs, lb, ub)
end

# convert arrays to vectors of svectors or scalars
to_svectors(x::AbstractVector{<:Union{Number,StaticArray}}) = x
to_svectors(x::AbstractVector{<:Number}, ::Val{1}) = SVector{1}.(x)
to_svectors(x::AbstractVector{<:SVector{N}}, ::Val{N}) where {N} = x
to_svectors(x::AbstractVector{<:AbstractVector{T}}, ::Val{N}=Val(length(first(x)))) where {T<:Number,N} =
    SVector{N,T}.(x)
to_svectors(x::AbstractMatrix{T}, ::Val{N}=Val(size(x,2))) where {T<:Number,N} =
    SVector{N,T}[row for row in eachrow(x)]
to_svectors(x::AbstractVector{<:AbstractArray{T}}, ::Val{S}=Val(size(first(x)))) where {T<:Number,S} =
    SArray{Tuple{S...},T,length(S),prod(S)}.(x)
to_svectors(x::AbstractArray{T}, ::Val{S}=Val(size(x)[2:end])) where {T<:Number,S} =
    SArray{Tuple{S...},T,length(S),prod(S)}[slice for slice in eachslice(x, dims=1)]
to_svectors(::AbstractArray{T,0}) where {T<:Number} =
    throw(ArgumentError("0-dimensional arrays not accepted"))

# normalize x and y arguments to vectors of svectors or scalars
chebregression(x::AbstractVecOrMat, y::AbstractArray, lb::AbstractVector, ub::AbstractVector, order::NTuple{N}) where {N} =
    chebregression(to_svectors(x, Val{N}()), to_svectors(y), SVector{N}(lb), SVector{N}(ub), order)

chebregression(x::AbstractVecOrMat, y::AbstractArray, order::NTuple{N}) where {N} =
    chebregression(to_svectors(x, Val{N}()), to_svectors(y), order)

# accept scalar bounds and order in 1d case
chebregression(x::AbstractVector{<:Real}, y::AbstractArray, lb::Real, ub::Real, order::Integer) =
    chebregression(x, y, SVector(lb), SVector(ub), (order,))
chebregression(x::AbstractVector{<:Real}, y::AbstractArray, order::Integer) =
    chebregression(x, y, minimum(x), maximum(x), order)

# construct lb and ub if omitted
chebregression(x::AbstractVector{<:SVector{N}}, y::AbstractVector, order::NTuple{N}) where {N} =
    chebregression(x, y, reduce((a,b) -> min.(a,b), x), reduce((a,b) -> max.(a,b), x), order)

# promote arguments to common types
function chebregression(x::AbstractVector{SVector{N,Tx}}, y::AbstractVector{Ty},
    lb::SVector{N,Tlb}, ub::SVector{N,Tub}, order::NTuple{N,<:Integer}) where {N,Tx<:Real,Tlb<:Real,Tub<:Real,Ty<:Union{StaticArray,Number}}
    Td = float(promote_type(Tx,Tub,Tlb))
    return chebregression(AbstractVector{SVector{N,Td}}(x), y, SVector{N,Td}(lb), SVector{N,Td}(ub), NTuple{N,Int}(order))
end

"""
    chebregression(x, y, [lb, ub,] order)

Return a Chebyshev polynomial (`ChebPoly`) constructed by
performing a least-square fit of Chebyshev polynomials of the
given `order`, where `x` are the coordinates of the data
points `y`.  `lb` and `ub` are the lower and upper bounds,
respectively, of the Chebyshev domain; these should normally
enclose all of the points in `x`, and default to the minimum
and maximum coordinates in `x` if they are omitted.

In the 1d case, `x` is an array of scalars, `lb < ub`
are scalars, and `order` is an integers.   In the `N`-dimensional
case, `order` is an `N`-tuple of integers (the order in each
dimension), `lb` and `ub` are `N`-component vectors, and
`x` is an array of `N`-component vectors (or a matrix with
`N` columns, interpreted as the vector components).

`y` can be a vector of numbers or a vector of vectors (for vector-
valued Chebyshev fits).  The latter case can also be input
as a matrix whose columns are the vector componnents.  `size(x,1)`
and `size(y,1)` must match, and must exceed `prod(order .+ 1)`.
"""
function chebregression end
