using Test, FastChebInterp, StaticArrays, Random, ChainRulesTestUtils

# similar to ≈, but acts elementwise on tuples
≈′(a::Tuple, b::Tuple; kws...) = length(a) == length(b) && all(xy -> isapprox(xy[1],xy[2]; kws...), zip(a,b))

Random.seed!(314159) # make chainrules tests deterministic

@testset "1d test" begin
    for T in (Float32, Float64)
        lb,ub = T(-0.3), T(0.9)
        f(x) = exp(x) / (1 + 2x^2)
        f′(x) = f(x) * (1 - 4x/(1 + 2x^2))
        @test_throws ArgumentError chebpoints(-1, lb, ub)
        x = chebpoints(48, lb, ub)
        @test eltype(x) == T
        interp = chebinterp(f.(x), lb, ub, tol=0)
        interp2 = chebinterp(f, 48, lb, ub, tol=0)
        @test interp isa FastChebInterp.ChebPoly{1,T,T}
        @test interp2.coefs ≈ interp.coefs
        @test repr("text/plain", interp) == "ChebPoly{1,$T,$T} order (48,) polynomial on [-0.3,0.9]"
        @test ndims(interp) == 1
        x1 = T(0.2)
        @test interp(x1) ≈ f(x1) ≈ interp2(x1)
        @test chebgradient(interp, x1) ≈′ (f(x1), f′(x1))
        test_frule(interp, x1, rtol=sqrt(eps(T)), atol=sqrt(eps(T)))
        test_rrule(interp, x1, rtol=sqrt(eps(T)), atol=sqrt(eps(T)))

        @test roots(chebinterp(x -> 1.1, 10, 0,1)) ≈ Float64[]
        @test roots(chebinterp(x -> x - 0.1, 10, 0,1)) ≈ [0.1]
        @test roots(chebinterp(x -> (x - 0.1)*(x-0.2), 10, 0,1)) ≈ [0.1, 0.2]
        @test roots(chebinterp(cos, 10000, 0, 1000))/pi ≈ (0:317) .+ T(0.5)
    end
end

@testset "2d test" begin
    for T in (Float32, Float64)
        lb, ub = T[-0.3,0.1], T[0.9,1.2]
        f(x) = exp(x[1]+2*x[2]) / (1 + 2x[1]^2 + x[2]^2)
        ∇f(x) = f(x) * SVector(1 - 4x[1]/(1 + 2x[1]^2 + x[2]^2), 2 - 2x[2]/(1 + 2x[1]^2 + x[2]^2))
        x = chebpoints((48,39), lb, ub)
        @test eltype(x) == SVector{2,T}
        interp = chebinterp(f.(x), lb, ub)
        interp2 = chebinterp(f, (48,39), lb, ub)
        @test interp isa FastChebInterp.ChebPoly{2,T,T}
        @test interp2.coefs ≈ interp.coefs
        interp0 = chebinterp(f.(x), lb, ub, tol=0)
        @test repr("text/plain", interp0) == "ChebPoly{2,$T,$T} order (48, 39) polynomial on [-0.3,0.9] × [0.1,1.2]"
        @test ndims(interp) == 2
        x1 = T[0.2, 0.8] # not too close to boundary or test_frule can fail by taking a big FD step
        @test interp(x1) ≈ f(x1)
        @test interp(x1) ≈ interp0(x1) rtol=10eps(T)
        @test all(n -> n[1] < n[2], zip(size(interp.coefs), size(interp0.coefs)))
        @test chebgradient(interp, x1) ≈′ (f(x1), ∇f(x1))
        test_frule(interp, x1, rtol=sqrt(eps(T)), atol=sqrt(eps(T)))
        test_rrule(interp, x1, rtol=sqrt(eps(T)), atol=sqrt(eps(T)))

        # univariate function in 2d should automatically drop down to univariate polynomial
        f1(x) = exp(x[1]) / (1 + 2x[1]^2)
        interp1 = chebinterp(f1.(x), lb, ub)
        @test interp1(x1) ≈ f1(x1)
        @test size(interp1.coefs, 2) == 1 # second dimension should have been dropped

        # complex and vector-valued interpolants:
        f2(x) = [f(x), cis(x[1]*x[2] + 2x[2])]
        ∇f2(x) = vcat(transpose(∇f(x)), transpose(SVector(im*x[2], im*(x[1] + 2)) * cis(x[1]*x[2] + 2x[2])))
        interp2 = chebinterp(f2.(x), lb, ub)
        @test interp2(x1) ≈ f2(x1)
        @test chebjacobian(interp2, x1) ≈′ (f2(x1), ∇f2(x1))
        test_frule(interp2, x1, rtol=sqrt(eps(T)), atol=sqrt(eps(T)))
        test_rrule(interp2, x1, rtol=sqrt(eps(T)), atol=sqrt(eps(T)))

        # chebinterp_v1
        av1 = Array{Complex{T}}(undef, 2, size(x)...)
        av1[1,:,:] .= f.(x)
        av1[2,:,:] .= (x -> f2(x)[2]).(x)
        interp2v1 = chebinterp_v1(av1, lb, ub)
        @test interp2v1(x1) ≈ f2(x1)
        @test chebjacobian(interp2v1, x1) ≈′ (f2(x1), ∇f2(x1))
    end
end

@testset "1d regression" begin
    x = range(1,2, length=200)
    y = @. sin.(x + 0.2) * x
    c = chebregression(x, y, 2)
    c2 = x.^(0:2)' \ y # Vandermonde-style fit
    @test c(1.1) ≈ c2[1] + 1.1 * c2[2] + 1.1^2 * c2[3] rtol=1e-13

    # test specialized Vandermonde matrix constructor
    x = [0.14, 0.95, 0.83, 0.13, 0.42, 0.12]
    xv = reinterpret(SVector{1,Float64}, x)
    A1 = FastChebInterp.chebvandermonde(x, 0.1, 0.99, 4)
    A = FastChebInterp._chebvandermonde(xv,  SVector(0.1), SVector(0.99), (4,))
    @test A ≈ A1 rtol=1e-13
    @test_throws ArgumentError FastChebInterp.chebvandermonde(x, 0.1, 0.9, 4)
    @test_throws ArgumentError FastChebInterp._chebvandermonde(xv,  SVector(0.1), SVector(0.9), (4,))
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

@testset "degree 1" begin
    lb,ub = -0.3, 0.9
    x = chebpoints(1, lb, ub)
    interp = chebinterp((x -> 2x+3).(x), lb, ub)
    @test interp(0) ≈ 3
    @test interp(0.1) ≈ 3.2
    @test chebgradient(interp, 0)[2] ≈ 2 ≈ chebgradient(interp, 0.1)[2]

    lb, ub = [-0.3,-0.1], [0.9,1.2]
    x = chebpoints((1,1), lb, ub)
    interp = chebinterp((x -> (2x[1]+3)*(4x[2]+5)).(x), lb, ub)
    @test interp([0,0]) ≈ 3*5
    @test interp([0.1,0]) ≈ 3.2*5
    @test interp([0,0.1]) ≈ 3*5.4
    @test chebgradient(interp, [0,0])[2] ≈ [2*5, 3*4]
end

@testset "degree 0" begin
    lb,ub = -0.3, 0.9
    x = chebpoints(0, lb, ub)
    interp = chebinterp(sin.(x), lb, ub)
    @test interp(0) ≈ interp(0.1) ≈ sin((lb+ub)/2)
    @test chebgradient(interp, 0)[2] == 0 == chebgradient(interp, 0.1)[2]

    lb, ub = [-0.3,-0.1], [0.9,1.2]
    x = chebpoints((0,0), lb, ub)
    interp = chebinterp((x -> sin(x[1])*cos(x[2])).(x), lb, ub)
    @test interp([0,0]) ≈ interp([0.1,0.2]) ≈ sin(0.3)*cos(0.55)
    @test chebgradient(interp, [0.1,0.2])[2] == [0, 0]

    lb, ub = [-0.3,-0.1], [0.9,1.2]
    x = chebpoints((1,0), lb, ub)
    interp = chebinterp((x -> (2x[1]+3)*cos(x[2])).(x), lb, ub)
    @test interp([0,0]) ≈ 3*cos(0.55)
    @test interp([0.1,0.22]) ≈ 3.2*cos(0.55)
    @test chebgradient(interp, [0.1,0.2])[2] == [2*cos(0.55), 0]
end

@testset "inference" begin
    @inferred FastChebInterp.droptol(rand(5,5), 0.0)
    @inferred chebinterp(rand(5,5), (0,0), (1,1))
end
