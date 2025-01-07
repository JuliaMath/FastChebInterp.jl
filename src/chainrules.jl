import ChainRulesCore
using ChainRulesCore: ProjectTo, NoTangent, @not_implemented

function ChainRulesCore.rrule(c::ChebPoly{1}, x::Real)
    project_x = ProjectTo(x)
    y, ∇y = chebgradient(c, x)
    chebpoly_pullback(∂y) = @not_implemented("no rrule for changes in ChebPoly itself"), project_x(real(∇y' * ∂y))
    y, chebpoly_pullback
end

function ChainRulesCore.rrule(c::ChebPoly, x::AbstractVector{<:Real})
    project_x = ProjectTo(x)
    y, J = chebjacobian(c, x)
    chebpoly_pullback(Δy) = @not_implemented("no rrule for changes in ChebPoly itself"), project_x(vec(real(J' * Δy)))
    y, chebpoly_pullback
end

ChainRulesCore.frule((Δself, Δx), c::ChebPoly{1}, x::Real) =
    ChainRulesCore.frule((Δself, SVector{1}(Δx)), c, SVector{1}(x))

function ChainRulesCore.frule((Δself, Δx), c::ChebPoly, x::AbstractVector)
    y, J = chebjacobian(c, x)
    if Δself isa ChainRulesCore.AbstractZero # Δself == 0
        Δy = J * Δx
        return y, y isa Number ? Δy[1] : Δy
    else # need derivatives with respect to changes in c
        # additional Δx from changes in bound:
        # --- recall x0 = @. (x - c.lb) * 2 / (c.ub - c.lb) - 1,
        #     but note that J already includes 2 / (c.ub - c.lb)
        d2 = @. (x - c.lb) / (c.ub - c.lb)
        Δx′ = @. Δx + (d2 - 1) * Δself.lb - d2 * Δself.ub
        Δy = J * Δx′

        # dependence on coefs is linear
        Δcoefs = typeof(c)(Δself.coefs, c.lb, c.ub, c.extrapolate)

        return y, (y isa Number ? Δy[1] : Δy) + Δcoefs(x)
    end
end
