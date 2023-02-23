using GsvdInitialization
using Test

using FileIO, LinearAlgebra, NMF

@testset "GsvdInitialization.jl" begin
    # Write your tests here.
    W, H = rand(10, 3), rand(3, 8)
    X = W*H

    W0, H0 = W, H
    Hadd = rand(2, 8)
    Wadd, a = GsvdInitialization.init_W(X, W0, H0, Hadd)
    @test a ≈ ones(size(W0, 2))
    @test sum(abs2, Wadd) <= 1e-12

    W0, H0 = zero(W), zero(H)
    U, S, V = svd(X)
    Hadd = V[:,1:3]'
    Wadd, a = GsvdInitialization.init_W(X, W0, H0, Hadd)
    @test sum(abs2, Wadd-(U*Diagonal(S))[:,1:3]) <= 1e-12

    W, H = load("test/WHtest1.jld2")["WH"]
    X = W*H
    # _W0, _H0 = NMF.randinit(X, 2)
    _W0, _H0 = NMF.nndsvd(X, 2, initdata = svd(X))
    X_nmf_0 = NMF.solve!(NMF.CoordinateDescent{Float64}(), X, _W0, _H0)
    W0, H0 = X_nmf_0.W, X_nmf_0.H
    kadd = 1
    W0_2, H0_2, cs, W0_1, H0_1, a, Wadd, Hadd = gsvdinit(X, W0, H0, kadd)
    X_nmf_1 = NMF.solve!(NMF.CoordinateDescent{Float64}(), X, copy(W0_2), copy(H0_2))
    W1, H1 = X_nmf_1.W, X_nmf_1.H
    save("test/WHtest_res_1.jld2", Dict("W0H0" => (W0,H0), "WaddHadd" => (Wadd,Hadd), "W0_1H0_1" => (W0_1,H0_1), "W0_2H0_2" => (W0_2,H0_2), "W1H1" => (W1,H1)))

end
