using GsvdInitialization
using Test
using Aqua
using Documenter
using ExplicitImports

using LinearAlgebra, NMF, FileIO
using OffsetArrays
using StableRNGs

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

DocMeta.setdocmeta!(GsvdInitialization, :DocTestSetup, :(using GsvdInitialization); recursive=true)
@testset "Doctests" begin
    doctest(GsvdInitialization; manual = false)
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
    rng = StableRNG(1)
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

    # `gsvdnmf(X, n2)` is sugar for `gsvdnmf(X, n2-1 => n2)`.  The two calls run
    # the same pipeline, so they agree only up to the ~1e-9 nondeterminism of
    # multithreaded BLAS in `nnmf`; `rtol = 1e-6` stays far tighter than the
    # `1e-4` NMF tol.
    X = rand(rng, 30, 20)
    result_1, _ = gsvdnmf(X, 10; alg=:cd)
    result_2, _ = gsvdnmf(X, 9 => 10; alg=:cd)
    W_gsvd_1, H_gsvd_1 = result_1.W, result_1.H
    W_gsvd_2, H_gsvd_2 = result_2.W, result_2.H
    @test isapprox(W_gsvd_1, W_gsvd_2; rtol = 1e-6)
    @test isapprox(H_gsvd_1, H_gsvd_2; rtol = 1e-6)

    # n2 == size(W, 2) is a caller bug: there is nothing to augment.  Reject it
    # eagerly rather than silently returning the input factorization.
    Wfit, Hfit = rand(rng, 30, 5), rand(rng, 5, 20)
    f = svd(X)
    @test_throws ArgumentError gsvdnmf(X, Wfit, Hfit, (f.U, f.S, f.V); n2 = 5)
    @test_throws "must be positive" gsvdnmf(X, Wfit, Hfit, (f.U, f.S, f.V); n2 = 5)
    @test_throws "must be positive" gsvdrecover(X, Wfit, Hfit, 0, (f.U, f.S, f.V))

    # An SVD with fewer components than `size(W0, 2)` cannot seed the
    # generalized SVD; reject it with a clear message rather than a BoundsError.
    fsmall = (f.U[:, 1:3], f.S[1:3], f.V[:, 1:3])
    @test_throws "has 3 components but size(W0, 2) = 5" gsvdrecover(X, Wfit, Hfit, 1, fsmall)

    # A single call augments by at most the current number of components.
    @test_throws "must be at most the initial number of components" gsvdnmf(X, Wfit, Hfit, (f.U, f.S, f.V); n2 = 11)
    @test_throws "must be at most size(W0, 2)" gsvdrecover(X, Wfit, Hfit, 6, (f.U, f.S, f.V))

    # `alg` must reach the polishing `nnmf`.  An unknown algorithm name is
    # rejected by `nnmf` only if `alg` is forwarded; the polishing call is the
    # sole `nnmf` invocation in this four-argument path.
    Waug, Haug = rand(rng, 30, 9), rand(rng, 9, 20)
    @test_throws "Invalid algorithm" gsvdnmf(X, Waug, Haug, (f.U, f.S, f.V); n2 = 10, alg = :__nonexistent__)
end

@testset "GsvdInitialization" begin
    rng = StableRNG(2)
    W, H = rand(rng, 10, 3), rand(rng, 3, 8)
    X = W*H
    U, S, V = svd(X)

    W0, H0 = copy(W), copy(H)
    Hadd = rand(rng, 2, 8)
    Wadd, a = GsvdInitialization.init_W(X, W0, H0, Hadd)
    @test a ≈ ones(size(W0, 2))
    @test norm(Wadd) <= 1e-8

    W0, H0 = zero(W), zero(H)
    Hadd = V[:,1:3]'
    Wadd, a = GsvdInitialization.init_W(X, W0, H0, Hadd)
    @test sum(abs2, Wadd-(U*Diagonal(S))[:,1:3]) <= 1e-12

    W0, H0 = rand(rng, 10, 4), rand(rng, 4, 8)
    Hadd = rand(rng, 2, 8)
    A, b, C, HH, γ = GsvdInitialization.obj_para(X, W0, H0, Hadd)
    a = rand(rng, 4)
    Wadd, a = GsvdInitialization.init_W(X, W0, H0, Hadd, α = a)
    E = a'*A*a+2*b'*a+C
    @test abs(E-sum(abs2, X-[repeat(a', size(W0, 1)).*W0 Wadd]*[H0;Hadd])) <= 1e-12

    β0 = rand(rng, 3)
    β = GsvdInitialization.Wcols_modification(X, repeat(β0', size(W, 1)).*W, H)
    @test β.*β0 ≈ ones(3)

    # When H0 is parallel to Hadd the Schur complement vanishes and A = 0, making
    # the QP degenerate.  init_W must return finite results rather than throwing
    # SingularException (which fnnls raises on Julia ≥ 1.12 for a zero pivot).
    W0_deg = rand(rng, Float64, 5, 1)
    H0_deg = rand(rng, Float64, 1, 8)
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
    rng = StableRNG(3)
    U = rand(rng, 10, 3)
    V = rand(rng, 3, 8)
    Xdense = U * V
    Xfact  = MockFactored(U, V)
    W0, H0 = rand(rng, 10, 4), rand(rng, 4, 8)
    Hadd   = rand(rng, 2, 8)
    fs     = svd(Xdense)
    f      = (fs.U, fs.S, fs.V)

    # `init_W` agrees across both X representations.
    Wadd_d, a_d = GsvdInitialization.init_W(Xdense, W0, H0, Hadd)
    Wadd_f, a_f = GsvdInitialization.init_W(Xfact,  W0, H0, Hadd)
    @test Wadd_d ≈ Wadd_f
    @test a_d    ≈ a_f

    # `Wcols_modification` likewise.
    β0 = rand(rng, 4)
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
    rng = StableRNG(4)
    W = rand(rng, Float32, 12, 4)
    H = rand(rng, Float32, 4, 9)
    X = W * H
    fs = svd(X)
    Wa, Ha, _ = gsvdrecover(X, copy(W), copy(H), 2, (fs.U, fs.S, fs.V))
    @test eltype(Wa) === Float32
    @test eltype(Ha) === Float32
end

@testset "integer-typed component counts" begin
    rng = StableRNG(5)
    X = rand(rng, 30, 20)
    # Non-`Int` integer types (e.g. `Int32`) must dispatch on the same methods
    # as plain `Int` — no `MethodError` from over-tight signatures.  Each pair
    # runs the same pipeline twice, so `rtol = 1e-6` for the same reason as the
    # cross-call checks in "test top wrapper".
    r_int,   _ = gsvdnmf(X, 9 => 10; alg = :cd)
    r_int32, _ = gsvdnmf(X, Int32(9) => Int32(10); alg = :cd)
    @test isapprox(r_int.W, r_int32.W; rtol = 1e-6)
    @test isapprox(r_int.H, r_int32.H; rtol = 1e-6)

    r_n2_int,   _ = gsvdnmf(X, 10; alg = :cd)
    r_n2_int32, _ = gsvdnmf(X, Int32(10); alg = :cd)
    @test isapprox(r_n2_int.W, r_n2_int32.W; rtol = 1e-6)

    # `gsvdrecover` likewise accepts a non-`Int` `kadd`.
    Wfit, Hfit = rand(rng, 30, 4), rand(rng, 4, 20)
    f = svd(X)
    Wa, Ha, _ = gsvdrecover(X, Wfit, Hfit, Int32(1), (f.U, f.S, f.V))
    @test size(Wa, 2) == 5
end

@testset "strategy dispatch" begin
    rng = StableRNG(6)
    X = rand(rng, 30, 20)
    # explicit `truncating` matches the no-strategy default.  These run the same
    # strategy, so they agree up to the ~1e-9 nondeterminism of multithreaded
    # BLAS in `nnmf`; `rtol = 1e-6` stays far tighter than the `1e-4` NMF tol.
    r_default, _ = gsvdnmf(X, 9 => 10; alg = :cd)
    r_explicit, _ = gsvdnmf(GsvdInitialization.truncating, X, 9 => 10; alg = :cd)
    @test isapprox(r_default.W, r_explicit.W; rtol = 1e-6)
    @test isapprox(r_default.H, r_explicit.H; rtol = 1e-6)

    # do-block form: anonymous strategy that simply forwards to `truncating`
    r_doblock, _ = gsvdnmf(X, 9 => 10; alg = :cd) do X0, W0, H0, Hadd
        GsvdInitialization.truncating(X0, W0, H0, Hadd)
    end
    @test isapprox(r_doblock.W, r_default.W; rtol = 1e-6)
    @test isapprox(r_doblock.H, r_default.H; rtol = 1e-6)
end

@testset "integer-n2 convenience methods and :multmse" begin
    rng = StableRNG(9)
    X = rand(rng, 30, 20)
    W, H = rand(rng, 30, 4), rand(rng, 4, 20)
    # `gsvdnmf(X, W, H, n2)` computes `tsvd(X, n2)` itself and forwards to the
    # explicit-`f` method; the explicit-strategy form runs the same pipeline,
    # so they agree to the usual cross-call rtol.
    r_default, _ = gsvdnmf(X, W, H, 5; alg = :cd)
    r_strategy, _ = gsvdnmf(GsvdInitialization.truncating, X, W, H, 5; alg = :cd)
    @test size(r_default.W, 2) == 5
    @test isapprox(r_default.W, r_strategy.W; rtol = 1e-6)
    @test isapprox(r_default.H, r_strategy.H; rtol = 1e-6)

    # :multmse floors the augmented factors to `truncmult` so multiplicative
    # updates (which cannot move entries off zero) can polish them.
    r_mult, _ = gsvdnmf(X, W, H, 5; alg = :multmse, maxiter = 10^4)
    @test size(r_mult.W, 2) == 5
    @test r_mult.converged
    @test all(>=(0), r_mult.W) && all(>=(0), r_mult.H)
    # A full-rank random X puts the best rank-5 residual near 0.13, so an
    # absolute bound is meaningless; require :multmse to land close to :cd.
    res_cd = sum(abs2, X - r_default.W * r_default.H)
    res_mult = sum(abs2, X - r_mult.W * r_mult.H)
    @test res_mult <= 1.1 * res_cd
end

@testset "generic axes" begin
    # The pipeline runs through `svd`, `nndsvd`, `nnmf`, and `sparse`, all of
    # which assume 1-based indexing, so the public entry points declare that
    # assumption with `require_one_based_indexing`.  Offset-axes inputs must
    # fail at entry with a clear error — without the declaration they fail deep
    # inside LinearAlgebra, or worse: an offset SVD wider than `size(W0, 2)`
    # makes the `1:n` factor slices succeed on the wrong columns, silently
    # returning wrong factors.
    rng = StableRNG(8)
    W, H = rand(rng, 10, 4), rand(rng, 4, 8)
    X = W * H + 0.01 * rand(rng, 10, 8)
    fs = svd(X)
    f = (fs.U, fs.S, fs.V)
    msg = "offset arrays are not supported"

    Xo = OffsetArray(X, -2, -3)
    Wo = OffsetArray(W, -2, 0)
    Ho = OffsetArray(H, 0, -3)
    # Full SVD of X has 8 components > size(W, 2) = 4: the silent-wrong-columns
    # shape.
    fo = (OffsetArray(fs.U, 0, -1), OffsetArray(fs.S, -1), OffsetArray(fs.V, 0, -1))

    @test_throws msg gsvdnmf(Xo, 3 => 4; alg = :cd)
    @test_throws msg gsvdnmf(X, Wo, Ho, f; n2 = 5)
    @test_throws msg gsvdrecover(X, Wo, Ho, 1, f)
    @test_throws msg gsvdrecover(Xo, W, H, 1, f)
    @test_throws msg gsvdrecover(X, W, H, 1, fo)
    Hadd = rand(rng, 1, 8)
    @test_throws msg GsvdInitialization.truncating(X, Wo, H, Hadd)
    @test_throws msg GsvdInitialization.joint_nnls(X, Wo, H, Hadd)

    # Lazy wrappers carry no axis shift; they must reproduce plain-input
    # results.
    vX, vW, vH = view(X, :, :), view(W, :, :), view(H, :, :)
    Wr, Hr, Λr = gsvdrecover(X, copy(W), copy(H), 2, f)
    Wv, Hv, Λv = gsvdrecover(vX, vW, vH, 2, f)
    @test Wv ≈ Wr
    @test Hv ≈ Hr
    @test Λv ≈ Λr
    Wjr, Hjr, _ = gsvdrecover(GsvdInitialization.joint_nnls, X, copy(W), copy(H), 2, f)
    Wjv, Hjv, _ = gsvdrecover(GsvdInitialization.joint_nnls, vX, vW, vH, 2, f)
    @test Wjv ≈ Wjr
    @test Hjv ≈ Hjr
    # `gsvdnmf` runs `nnmf`, so two runs of the same pipeline agree only to the
    # ~1e-9 nondeterminism of multithreaded BLAS; `rtol = 1e-6` as in the
    # cross-call checks above.
    r_plain, _ = gsvdnmf(X, copy(W), copy(H), f; n2 = 5, alg = :cd)
    r_view, _  = gsvdnmf(vX, vW, vH, f; n2 = 5, alg = :cd)
    @test isapprox(r_plain.W, r_view.W; rtol = 1e-6)
    @test isapprox(r_plain.H, r_view.H; rtol = 1e-6)
end

@testset "joint optimize W and alpha" begin
    rng = StableRNG(7)
    W = W_GT
    H = H_GT
    X = W*H
    result_joint, _ = gsvdnmf(GsvdInitialization.joint_nnls, X, 9=>10; alg = :cd, maxiter = 10^5, tol_final=1e-4, tol_intermediate = 1e-4);
    W_gsvd, H_gsvd = result_joint.W, result_joint.H
    @test size(W_gsvd, 2) == 10
    @test sum(abs2, X-W_gsvd*H_gsvd)/sum(abs2, X) < 2e-10

    W, H = rand(rng, 10, 3), rand(rng, 3, 8)
    X = W*H
    U, S, V = svd(X)

    W0, H0 = copy(W), copy(H)
    Hadd = rand(rng, 2, 8)
    Wadd, a = GsvdInitialization.init_W_joint_nnls(X, W0, H0, Hadd)
    @test a ≈ ones(size(W0, 2))
    @test norm(Wadd) <= 1e-8

    G = GsvdInitialization.gram_sp_C(W0, H0, Hadd)[1]
    b = GsvdInitialization.gram_b(X, W0, H0, Hadd)
    Wadd = rand(rng, 10, 2)
    α = rand(rng, 3)
    θ = vcat(vec(Wadd), α)
    E = θ'*G*θ-2*b'*θ+sum(abs2, X)
    @test abs(E-sum(abs2, X-[repeat(α', size(W0, 1)).*W0 Wadd]*[H0;Hadd])) <= 1e-12

end
