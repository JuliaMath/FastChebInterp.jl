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
    interp0 = chebfit(f.(x), lb, ub, tol=0)
    @test ndims(interp) == 2
    x1 = [0.2, 0.3]
    @test interp(x1) ≈ f(x1)
    @test interp(x1) ≈ interp0(x1) rtol=1e-15
    @test all(n -> n[1] < n[2], zip(size(interp.coefs), size(interp0.coefs)))
    @test chebgradient(interp, x1) ≈′ (f(x1), ∇f(x1))

    # univariate function in 2d should automatically drop down to univariate polynomial
    f1(x) = exp(x[1]) / (1 + 2x[1]^2)
    interp1 = chebfit(f1.(x), lb, ub)
    @test interp1(x1) ≈ f1(x1)
    @test size(interp1.coefs, 2) == 1 # second dimension should have been dropped

    # complex and vector-valued interpolants:
    f2(x) = [f(x), cis(x[1]*x[2] + 2x[2])]
    ∇f2(x) = vcat(transpose(∇f(x)), transpose(SVector(im*x[2], im*(x[1] + 2)) * cis(x[1]*x[2] + 2x[2])))
    interp2 = chebfit(f2.(x), lb, ub)
    @test interp2(x1) ≈ f2(x1)
    @test chebjacobian(interp2, x1) ≈′ (f2(x1), ∇f2(x1))

    # chebfitv1
    av1 = Array{ComplexF64}(undef, 2, size(x)...)
    av1[1,:,:] .= f.(x)
    av1[2,:,:] .= (x -> f2(x)[2]).(x)
    interp2v1 = chebfitv1(av1, lb, ub)
    @test interp2v1(x1) ≈ f2(x1)
    @test chebjacobian(interp2v1, x1) ≈′ (f2(x1), ∇f2(x1))
end

@testset "1d regression" begin
    x = range(1,2, length=200)
    y = @. sin.(x + 0.2) * x
    c = chebregression(x, y, 2)
    c2 = x.^(0:2)' \ y # Vandermonde-style fit
    @test c(1.1) ≈ c2[1] + 1.1 * c2[2] + 1.1^2 * c2[3] rtol=1e-13
end

@testset "2d regression" begin
    x = vec(SVector.(range(-3, 2, length=50), range(0,1, length=40)'))
    fr(x::SVector{2}) = sin(x[1]/3 + 1)*exp(x[2])
    c = chebregression(x, fr.(x), (2,3))

    # Vandermonde-style fit:
    powsvec(x) = vec([x[1]^i * x[2]^j for i=0:2, j=0:3])'
    c2 = mapreduce(powsvec, vcat, x) \ fr.(x)

    p = SVector(1.12345, sqrt(0.5))
    @test c(p) ≈ powsvec(p)*c2 rtol=1e-13

    cM = chebregression(mapreduce(transpose∘Vector, vcat, x), fr.(x), (2,3))
    @test cM.coefs == c.coefs

    fr4(x::SVector{2}) = cos(x[1]/3 + 1)*sin(x[2])
    c4 = chebregression(x, fr4.(x), (2,3))
    F14 = [SVector(fr(x),fr4(x)) for x in x]
    c14 = chebregression(x, F14, (2,3))
    @test first.(c14.coefs) ≈ c.coefs rtol=1e-13
    @test last.(c14.coefs) ≈ c4.coefs rtol=1e-13
    c14M = chebregression(x, mapreduce(transpose∘Vector, vcat, F14), (2,3))
    @test c14M.coefs == c14.coefs
end
