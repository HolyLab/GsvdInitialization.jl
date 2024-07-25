using GsvdInitialization
using Test

using LinearAlgebra, NMF

include(joinpath(pwd(), "demo/generate_ground_truth.jl"))

W_GT, H_GT = generate_ground_truth()

@testset "test top wrapper" begin
    W = W_GT
    H = H_GT
    X = W*H
    standard_nmf = nnmf(float(X), 10; init=:nndsvd, tol=1e-4, initdata = svd(float(X)))
    W_gsvd, H_gsvd = gsvdnmf(float(X), 9=>10; alg = :cd, maxiter = 10^5, tol_final=1e-4, tol_intermediate = 1e-4);
    img_tol_int = sum(abs2, X)
    @test size(W_gsvd, 2) == 10
    @test sum(abs2, X-standard_nmf.W*standard_nmf.H)/sum(abs2, X) > sum(abs2, X-W_gsvd*H_gsvd)/sum(abs2, X)
    @test sum(abs2, X-W_gsvd*H_gsvd)/sum(abs2, X) < 2e-10

    X = rand(30, 20)
    W_gsvd_1, H_gsvd_1 = gsvdnmf(X, 10; alg=:cd)
    W_gsvd_2, H_gsvd_2 = gsvdnmf(X, 9 => 10; alg=:cd)
    @test sum(abs2, W_gsvd_1-W_gsvd_2) <= 1e-12
    @test sum(abs2, H_gsvd_1-H_gsvd_2) <= 1e-12
end

@testset "GsvdInitialization.jl" begin
    W, H = rand(10, 3), rand(3, 8)
    X = W*H
    U, S, V = svd(X)

    W0, H0 = copy(W), copy(H)
    Hadd = rand(2, 8)
    Wadd, a = GsvdInitialization.init_W(X, W0, H0, Hadd)
    @test a ≈ ones(size(W0, 2))
    @test norm(Wadd) <= 1e-8
    
    W0, H0 = zero(W), zero(H)
    Hadd = V[:,1:3]'
    Wadd, a = GsvdInitialization.init_W(X, W0, H0, Hadd)
    @test sum(abs2, Wadd-(U*Diagonal(S))[:,1:3]) <= 1e-12

    W0, H0 = rand(10, 4), rand(4, 8)
    Hadd = rand(2, 8)
    A, b, C, HH, γ = GsvdInitialization.obj_para(X, W0, H0, Hadd)
    a = rand(4)
    Wadd, a = GsvdInitialization.init_W(X, W0, H0, Hadd, α = a)
    E = a'*A*a+2*b'*a+C
    @test abs(E-sum(abs2, X-[repeat(a', size(W0, 1)).*W0 Wadd]*[H0;Hadd])) <= 1e-12

    β0 = rand(3)
    β = GsvdInitialization.Wcols_modification(X, repeat(β0', size(W, 1)).*W, H)
    @test β.*β0 ≈ ones(3)

end
