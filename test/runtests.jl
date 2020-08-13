using Test, FastChebInterp

@testset "2d test" begin
    lb, ub = [-0.3,0.1], [0.9,1.2]
    f(x) = exp(x[1]+2*x[2]) / (1 + 2x[1]^2 + x[2]^2)
    x = chebpoints((48,39), lb, ub)
    interp = chebinterp(f.(x), lb, ub)
    @test ndims(interp) == 2
    x1 = [0.2, 0.3]
    @test interp(x1) ≈ f(x1)

    # complex and SVector-valued interpolants:
    f2(x) = SVector(f(x), cis(x[1]*x[2] + 2x[2]))
    interp2 = chebinterp(f2.(x), lb, ub)
    @test interp2(x1) ≈ f2(x1)
end
