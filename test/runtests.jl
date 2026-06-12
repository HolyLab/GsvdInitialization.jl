using GsvdInitialization
using Test
using Aqua
using ExplicitImports

using LinearAlgebra, NMF, FileIO

@testset "Aqua" begin
    Aqua.test_all(GsvdInitialization)
end

@testset "ExplicitImports" begin
    # `nndsvd` is exported by NMF but not declared `public` in NMF.jl.
    test_explicit_imports(GsvdInitialization;
                          ignore = (:nndsvd,),
                          all_explicit_imports_are_public = VERSION >= v"1.11",
                          all_qualified_accesses_are_public = VERSION >= v"1.11")
end

# Minimal `Factorization` subtype that implements just the matrix products and
# `sum(abs2, ·)` that `gsvdrecover` calls on `X`.  Used to verify that
# `gsvdrecover` and its helpers accept any `X` for which those operations
# work — *not* only `AbstractArray` — so that callers can pass low-rank
# representations (e.g. `FactoredMatrices.FactoredMatrix`) without ever
# materializing the dense product.  Deliberately not `<: AbstractMatrix` so a
# regression that tightens the signatures back to `AbstractArray{T}` fails this
# test rather than silently materializing via `getindex` fallback.
struct MockFactored{T} <: LinearAlgebra.Factorization{T}
    U::Matrix{T}
    V::Matrix{T}
end
Base.size(F::MockFactored) = (size(F.U, 1), size(F.V, 2))
Base.size(F::MockFactored, d) = d == 1 ? size(F.U, 1) : (d == 2 ? size(F.V, 2) : 1)
Base.:*(F::MockFactored, A::AbstractMatrix) = F.U * (F.V * A)
Base.:*(A::AbstractMatrix, F::MockFactored) = (A * F.U) * F.V
Base.sum(::typeof(abs2), F::MockFactored) = sum((F.U' * F.U) .* (F.V * F.V'))
# `joint_nnls`'s `truncatepos` step needs `X - W*H`; materializing densely here
# is fine — the point is that no `AbstractArray` constraint rejects `X`.
Base.:-(F::MockFactored, A::AbstractMatrix) = F.U * F.V - A

include(joinpath(dirname(@__DIR__), "demo/generate_ground_truth.jl"))

W_GT, H_GT = generate_ground_truth()
svdX = load_svd_of_gt()

@testset "test top wrapper" begin
    W = W_GT
    H = H_GT
    X = W*H
    standard_nmf = nnmf(X, 10; alg = :cd, init=:nndsvd, tol=1e-4, maxiter = 10^5, initdata = svdX)
    result_gsvd, Λ_gsvd = gsvdnmf(X, 9=>10; alg = :cd, maxiter = 10^5, tol_final=1e-4, tol_intermediate = 1e-4);
    W_gsvd, H_gsvd = result_gsvd.W, result_gsvd.H
    @test size(W_gsvd, 2) == 10
    @test sum(abs2, X-W_gsvd*H_gsvd)/sum(abs2, X) < 2e-10
    @test sum(abs2, X-standard_nmf.W*standard_nmf.H)/sum(abs2, X) > sum(abs2, X-W_gsvd*H_gsvd)/sum(abs2, X)
    @test length(Λ_gsvd) == 9

    X = rand(30, 20)
    result_1, _ = gsvdnmf(X, 10; alg=:cd)
    result_2, _ = gsvdnmf(X, 9 => 10; alg=:cd)
    W_gsvd_1, H_gsvd_1 = result_1.W, result_1.H
    W_gsvd_2, H_gsvd_2 = result_2.W, result_2.H
    @test sum(abs2, W_gsvd_1-W_gsvd_2) <= 1e-12
    @test sum(abs2, H_gsvd_1-H_gsvd_2) <= 1e-12

    # n2 == size(W, 2) is a caller bug: there is nothing to augment.  Reject it
    # eagerly rather than silently returning the input factorization.
    Wfit, Hfit = rand(30, 5), rand(5, 20)
    f = svd(X)
    @test_throws ArgumentError gsvdnmf(X, Wfit, Hfit, (f.U, f.S, f.V); n2 = 5)
    @test_throws "must be positive" gsvdnmf(X, Wfit, Hfit, (f.U, f.S, f.V); n2 = 5)
    @test_throws "must be positive" gsvdrecover(X, Wfit, Hfit, 0, (f.U, f.S, f.V))

    # An SVD with fewer components than `size(W0, 2)` cannot seed the
    # generalized SVD; reject it with a clear message rather than a BoundsError.
    fsmall = (f.U[:, 1:3], f.S[1:3], f.V[:, 1:3])
    @test_throws "has 3 components but size(W0, 2) = 5" gsvdrecover(X, Wfit, Hfit, 1, fsmall)

    # `alg` must reach the polishing `nnmf`.  An unknown algorithm name is
    # rejected by `nnmf` only if `alg` is forwarded; the polishing call is the
    # sole `nnmf` invocation in this four-argument path.
    Waug, Haug = rand(30, 9), rand(9, 20)
    @test_throws "Invalid algorithm" gsvdnmf(X, Waug, Haug, (f.U, f.S, f.V); n2 = 10, alg = :__nonexistent__)
end

@testset "GsvdInitialization" begin
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

    # When H0 is parallel to Hadd the Schur complement vanishes and A = 0, making
    # the QP degenerate.  init_W must return finite results rather than throwing
    # SingularException (which fnnls raises on Julia ≥ 1.12 for a zero pivot).
    W0_deg = rand(Float64, 5, 1)
    H0_deg = rand(Float64, 1, 8)
    X_deg  = W0_deg * H0_deg
    Hadd_deg = H0_deg   # parallel to H0 → Schur complement = 0 → A = 0
    Wadd_deg, a_deg = GsvdInitialization.init_W(X_deg, W0_deg, H0_deg, Hadd_deg)
    @test all(isfinite, Wadd_deg)
    @test all(isfinite, a_deg)
end

# `gsvdrecover` and its helpers accept any `X` for which `X * Y`, `Y * X`, and
# `sum(abs2, X)` are defined.  Passing a non-`AbstractArray` `X` (e.g. a
# `LinearAlgebra.Factorization` subtype that stores a low-rank form) lets the
# dense product never be materialized, which matters when `X` would otherwise
# be the largest array per call.
@testset "non-AbstractArray X" begin
    U = rand(10, 3)
    V = rand(3, 8)
    Xdense = U * V
    Xfact  = MockFactored(U, V)
    W0, H0 = rand(10, 4), rand(4, 8)
    Hadd   = rand(2, 8)
    fs     = svd(Xdense)
    f      = (fs.U, fs.S, fs.V)

    # `init_W` agrees across both X representations.
    Wadd_d, a_d = GsvdInitialization.init_W(Xdense, W0, H0, Hadd)
    Wadd_f, a_f = GsvdInitialization.init_W(Xfact,  W0, H0, Hadd)
    @test Wadd_d ≈ Wadd_f
    @test a_d    ≈ a_f

    # `Wcols_modification` likewise.
    β0 = rand(4)
    W_scaled = repeat(β0', size(W0, 1)) .* W0
    @test GsvdInitialization.Wcols_modification(Xdense, W_scaled, H0) ≈
          GsvdInitialization.Wcols_modification(Xfact,  W_scaled, H0)

    # End-to-end `gsvdrecover` agrees on the components it returns.
    Wd, Hd, _ = GsvdInitialization.gsvdrecover(Xdense, copy(W0), copy(H0), 2, f)
    Wf, Hf, _ = GsvdInitialization.gsvdrecover(Xfact,  copy(W0), copy(H0), 2, f)
    @test Wd ≈ Wf
    @test Hd ≈ Hf

    # The `joint_nnls` strategy also accepts a factored `X` (it needs `X - W*H`
    # and `eltype(X)` on top of `*`/`sum(abs2, ·)`).
    Wjd, ajd = GsvdInitialization.init_W_joint_nnls(Xdense, W0, H0, Hadd)
    Wjf, ajf = GsvdInitialization.init_W_joint_nnls(Xfact,  W0, H0, Hadd)
    @test Wjd ≈ Wjf
    @test ajd ≈ ajf

    Wd_j, Hd_j, _ = GsvdInitialization.gsvdrecover(GsvdInitialization.joint_nnls, Xdense, copy(W0), copy(H0), 2, f)
    Wf_j, Hf_j, _ = GsvdInitialization.gsvdrecover(GsvdInitialization.joint_nnls, Xfact,  copy(W0), copy(H0), 2, f)
    @test Wd_j ≈ Wf_j
    @test Hd_j ≈ Hf_j
end

@testset "eltype genericity" begin
    # The default (`truncating`) path must preserve the input eltype rather than
    # silently promoting to `Float64`.
    W = rand(Float32, 12, 4)
    H = rand(Float32, 4, 9)
    X = W * H
    fs = svd(X)
    Wa, Ha, _ = gsvdrecover(X, copy(W), copy(H), 2, (fs.U, fs.S, fs.V))
    @test eltype(Wa) === Float32
    @test eltype(Ha) === Float32
end

@testset "integer-typed component counts" begin
    X = rand(30, 20)
    # Non-`Int` integer types (e.g. `Int32`) must dispatch on the same methods
    # as plain `Int` — no `MethodError` from over-tight signatures.
    r_int,   _ = gsvdnmf(X, 9 => 10; alg = :cd)
    r_int32, _ = gsvdnmf(X, Int32(9) => Int32(10); alg = :cd)
    @test r_int.W ≈ r_int32.W
    @test r_int.H ≈ r_int32.H

    r_n2_int,   _ = gsvdnmf(X, 10; alg = :cd)
    r_n2_int32, _ = gsvdnmf(X, Int32(10); alg = :cd)
    @test r_n2_int.W ≈ r_n2_int32.W

    # `gsvdrecover` likewise accepts a non-`Int` `kadd`.
    Wfit, Hfit = rand(30, 4), rand(4, 20)
    f = svd(X)
    Wa, Ha, _ = gsvdrecover(X, Wfit, Hfit, Int32(1), (f.U, f.S, f.V))
    @test size(Wa, 2) == 5
end

@testset "strategy dispatch" begin
    X = rand(30, 20)
    # explicit `truncating` matches the no-strategy default
    r_default, _ = gsvdnmf(X, 9 => 10; alg = :cd)
    r_explicit, _ = gsvdnmf(GsvdInitialization.truncating, X, 9 => 10; alg = :cd)
    @test r_default.W ≈ r_explicit.W
    @test r_default.H ≈ r_explicit.H

    # do-block form: anonymous strategy that simply forwards to `truncating`
    r_doblock, _ = gsvdnmf(X, 9 => 10; alg = :cd) do X0, W0, H0, Hadd
        GsvdInitialization.truncating(X0, W0, H0, Hadd)
    end
    @test r_doblock.W ≈ r_default.W
    @test r_doblock.H ≈ r_default.H
end

@testset "joint optimize W and alpha" begin
    W = W_GT
    H = H_GT
    X = W*H
    result_joint, _ = gsvdnmf(GsvdInitialization.joint_nnls, X, 9=>10; alg = :cd, maxiter = 10^5, tol_final=1e-4, tol_intermediate = 1e-4);
    W_gsvd, H_gsvd = result_joint.W, result_joint.H
    @test size(W_gsvd, 2) == 10
    @test sum(abs2, X-W_gsvd*H_gsvd)/sum(abs2, X) < 2e-10

    W, H = rand(10, 3), rand(3, 8)
    X = W*H
    U, S, V = svd(X)

    W0, H0 = copy(W), copy(H)
    Hadd = rand(2, 8)
    Wadd, a = GsvdInitialization.init_W_joint_nnls(X, W0, H0, Hadd)
    @test a ≈ ones(size(W0, 2))
    @test norm(Wadd) <= 1e-8

    G = GsvdInitialization.gram_sp_C(W0, H0, Hadd)[1]
    b = GsvdInitialization.gram_b(X, W0, H0, Hadd)
    Wadd = rand(10, 2)
    α = rand(3)
    θ = vcat(vec(Wadd), α)
    E = θ'*G*θ-2*b'*θ+sum(abs2, X)
    @test abs(E-sum(abs2, X-[repeat(α', size(W0, 1)).*W0 Wadd]*[H0;Hadd])) <= 1e-12

end
