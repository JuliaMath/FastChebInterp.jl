using Test, FastChebInterp, StaticArrays

# similar to ≈, but acts elementwise on tuples
≈′(a::Tuple, b::Tuple; kws...) where {N} = length(a) == length(b) && all(xy -> isapprox(xy[1],xy[2]; kws...), zip(a,b))

@testset "1d test" begin
    lb,ub = -0.3, 0.9
    f(x) = exp(x) / (1 + 2x^2)
    f′(x) = f(x) * (1 - 4x/(1 + 2x^2))
    x = chebpoints(48, lb, ub)
    interp = chebfit(f.(x), lb, ub)
    @test ndims(interp) == 1
    x1 = 0.2
    @test interp(x1) ≈ f(x1)
    @test chebgradient(interp, x1) ≈′ (f(x1), f′(x1))
end

@testset "2d test" begin
    lb, ub = [-0.3,0.1], [0.9,1.2]
    f(x) = exp(x[1]+2*x[2]) / (1 + 2x[1]^2 + x[2]^2)
    ∇f(x) = f(x) * SVector(1 - 4x[1]/(1 + 2x[1]^2 + x[2]^2), 2 - 2x[2]/(1 + 2x[1]^2 + x[2]^2))
    x = chebpoints((48,39), lb, ub)
    interp = chebfit(f.(x), lb, ub)
    @test ndims(interp) == 2
    x1 = [0.2, 0.3]
    @test interp(x1) ≈ f(x1)
    @test chebgradient(interp, x1) ≈′ (f(x1), ∇f(x1))

    # complex and vector-valued interpolants:
    f2(x) = [f(x), cis(x[1]*x[2] + 2x[2])]
    ∇f2(x) = vcat(transpose(∇f(x)), transpose(SVector(im*x[2], im*(x[1] + 2)) * cis(x[1]*x[2] + 2x[2])))
    interp2 = chebfit(f2.(x), lb, ub)
    @test interp2(x1) ≈ f2(x1)
    @test chebjacobian(interp2, x1) ≈′ (f2(x1), ∇f2(x1))
end
