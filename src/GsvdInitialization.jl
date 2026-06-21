"""
`GsvdInitialization` augments an existing low-rank non-negative matrix
factorization (NMF) `X ≈ W*H` with additional components, using the
generalized singular value decomposition (GSVD) between the current
factorization and an SVD of `X` to discover what the factorization is missing
(the GSVD-NMF method, [doi:10.1016/j.isci.2026.114708](https://doi.org/10.1016/j.isci.2026.114708)).

The main entry points are [`gsvdnmf`](@ref), which augments a factorization
and polishes the result with NMF, and [`gsvdrecover`](@ref), which performs
the augmentation step alone.
"""
module GsvdInitialization

using LinearAlgebra: Diagonal, I, Symmetric, UpperTriangular, cholesky, diag, isposdef, svd, tr
using NMF: NMF, nnmf, nndsvd
using TSVD: tsvd
using NonNegLeastSquares: nonneg_lsq
using Kronecker: kronecker
using SparseArrays: sparse

export gsvdnmf,
    gsvdrecover

@static if VERSION >= v"1.11"
    eval(Meta.parse("public appending, truncating, joint_nnls"))
end

"""
    result, Λ = gsvdnmf([strategy,] X::AbstractMatrix, W::AbstractMatrix, H::AbstractMatrix, f;
                        n2 = size(first(f), 2), tol_final = 1e-4,
                        alg = :cd, truncmult = 1e-5, kwargs...)

Augment the existing factorization `X ≈ W*H` to `n2` components, then polish
the result with NMF. An integer `n2` may be passed in place of `f`, in which
case a truncated SVD with `n2` components is computed internally.

Return the `NMF.Result` of the polishing run (its `W` and `H` fields hold the
augmented factors) and `Λ`, the generalized singular values that ranked the
candidate augmentation directions.

See [`gsvdrecover`](@ref) for the augmentation step alone, without polishing.

# Arguments

- `strategy`: a callable
  `(X, W0, H0, Hadd, Λ, kadd) -> (W_augmented, H_augmented)` that assembles
  fully non-negative augmented factors from the candidate directions (see
  [`gsvdrecover`](@ref) for the full contract). Defaults to
  [`GsvdInitialization.truncating`](@ref);
  [`GsvdInitialization.joint_nnls`](@ref) and
  [`GsvdInitialization.appending`](@ref) are the alternative bundled
  strategies. The polishing run is sized to the number of components the
  strategy actually kept, so a strategy that adds fewer than `n2 - n1` (e.g.
  [`appending`](@ref)) yields a final factorization with fewer than `n2`
  components.
- `X`: non-negative data matrix.
- `W`: left factor of the existing factorization, of size `(m, n1)`.
- `H`: right factor of the existing factorization, of size `(n1, p)`.
- `f`: a singular value decomposition of `X` with at least `n1` components,
  e.g. from `LinearAlgebra.svd` or `TSVD.tsvd`. Any object whose factors `U`,
  `S`, `V` are indexable as `f[1]`, `f[2]`, `f[3]` works.

All array arguments, including the factors of `f`, must use 1-based indexing.

# Keyword arguments

- `n2`: the number of components after augmentation; `n2 - n1` must satisfy
  `1 ≤ n2 - n1 ≤ n1` (at most a doubling per call).
- `tol_final`: convergence tolerance of the NMF polishing run (default: `1e-4`).
- `alg`: NMF algorithm for the polishing run, forwarded to `NMF.nnmf`
  (default: `:cd`). With `alg == :multmse`, the augmented factors are first
  floored at `truncmult`, because multiplicative updates require strictly
  positive factors.
- `truncmult`: flooring level applied when `alg == :multmse` (default: `1e-5`).

Remaining keyword arguments are forwarded to `NMF.nnmf`.

# Examples

```jldoctest
julia> using LinearAlgebra: svd

julia> X = Float64[1 0 0 1 0; 0 1 0 1 1; 0 0 1 0 1; 1 1 1 2 2];  # rank 3

julia> W0 = Float64[1 0; 0 1; 0 0; 1 1]; H0 = Float64[1 0 0 1 0; 0 1 0 1 1];  # rank-2 factorization of X

julia> result, Λ = gsvdnmf(X, W0, H0, svd(X); n2 = 3);

julia> size(result.W), size(result.H)
((4, 3), (3, 5))
```
"""
function gsvdnmf(
        strategy, X::AbstractMatrix, W::AbstractMatrix, H::AbstractMatrix, f;
        n2 = size(first(f), 2),
        tol_final = 1.0e-4,
        alg = :cd,
        truncmult = 1.0e-5,
        kwargs...
    )
    Base.require_one_based_indexing(X, W, H)
    n1 = size(W, 2)
    kadd = n2 - n1
    kadd > 0 || throw(ArgumentError("The number of components to add must be positive; got n2 = $n2, size(W, 2) = $n1"))
    kadd <= n1 || throw(ArgumentError("The number of components to add must be at most the initial number of components; got kadd = $kadd, size(W, 2) = $n1"))
    size(first(f), 2) >= n1 || throw(ArgumentError("The supplied SVD has $(size(first(f), 2)) components but size(W, 2) = $n1 are required"))
    W_recover, H_recover, Λ = gsvdrecover(strategy, X, copy(W), copy(H), kadd, f)
    if alg == :multmse
        W_recover, H_recover = max.(W_recover, truncmult), max.(H_recover, truncmult)
    end
    # An "at most kadd" strategy (e.g. `appending`) may keep fewer than
    # `n2 - n1` components; size the polishing run to what it kept.
    result_recover = nnmf(X, size(W_recover, 2); kwargs..., alg, init = :custom, tol = tol_final, W0 = copy(W_recover), H0 = copy(H_recover))
    return result_recover, Λ
end
gsvdnmf(X::AbstractMatrix, W::AbstractMatrix, H::AbstractMatrix, f; kwargs...) =
    gsvdnmf(truncating, X, W, H, f; kwargs...)
gsvdnmf(strategy, X::AbstractMatrix, W::AbstractMatrix, H::AbstractMatrix, n2::Integer; kwargs...) =
    gsvdnmf(strategy, X, W, H, tsvd(X, n2); kwargs...)
gsvdnmf(X::AbstractMatrix, W::AbstractMatrix, H::AbstractMatrix, n2::Integer; kwargs...) =
    gsvdnmf(truncating, X, W, H, n2; kwargs...)

"""
    result, Λ = gsvdnmf([strategy,] X::AbstractMatrix, ncomponents;
                        tol_final = 1e-4, tol_intermediate = tol_final, kwargs...)

Perform GSVD-NMF on the non-negative data matrix `X`: compute an NMF with `n1`
components, augment it to `n2` components, and polish with a final NMF run.

The initial factorization is computed by `NMF.nnmf` with NNDSVD initialization
seeded from a truncated SVD of `X`; the same SVD supplies the augmentation
directions. To augment an existing factorization instead, use the four-argument
[`gsvdnmf`](@ref) method.

Return the `NMF.Result` of the final NMF run (its `W` and `H` fields hold the
factors) and `Λ`, the generalized singular values that ranked the candidate
augmentation directions.

# Arguments

- `strategy`: see the four-argument [`gsvdnmf`](@ref) method. Defaults to
  [`GsvdInitialization.truncating`](@ref).
- `X`: non-negative data matrix; must use 1-based indexing.
- `ncomponents`: a pair `n1 => n2` of integers requesting augmentation from
  `n1` to `n2` components, where `n1 < n2 ≤ 2n1`. An integer `n` is shorthand
  for `n-1 => n` (add a single component).

# Keyword arguments

- `tol_final`: convergence tolerance of the final NMF run (default: `1e-4`).
- `tol_intermediate`: convergence tolerance of the initial rank-`n1` NMF run
  (default: same as `tol_final`).

Remaining keyword arguments are forwarded to `NMF.nnmf`.

# Examples

```jldoctest
julia> X = Float64[1 0 0 1 0; 0 1 0 1 1; 0 0 1 0 1; 1 1 1 2 2];  # rank 3

julia> result, Λ = gsvdnmf(X, 2 => 3);

julia> size(result.W), size(result.H)
((4, 3), (3, 5))

julia> sum(abs2, X - result.W*result.H) < 1e-6 * sum(abs2, X)  # near-exact rank-3 fit
true
```
"""
function gsvdnmf(strategy, X::AbstractMatrix, ncomponents::Pair{<:Integer, <:Integer}; tol_final = 1.0e-4, tol_intermediate = tol_final, kwargs...)
    Base.require_one_based_indexing(X)
    n1, n2 = ncomponents
    f = tsvd(X, n2)
    W0, H0 = nndsvd(X, n1; initdata = (U = f[1], S = f[2], V = f[3]))
    result_initial_nmf = nnmf(X, n1; kwargs..., init = :custom, tol = tol_intermediate, W0 = copy(W0), H0 = copy(H0))
    W_initial_nmf, H_initial_nmf = result_initial_nmf.W, result_initial_nmf.H
    return gsvdnmf(strategy, X, W_initial_nmf, H_initial_nmf, f; kwargs..., n2, tol_final)
end
gsvdnmf(X::AbstractMatrix, ncomponents::Pair{<:Integer, <:Integer}; kwargs...) =
    gsvdnmf(truncating, X, ncomponents; kwargs...)
gsvdnmf(strategy, X::AbstractMatrix, ncomponents_final::Integer; kwargs...) =
    gsvdnmf(strategy, X, ncomponents_final - 1 => ncomponents_final; kwargs...)
gsvdnmf(X::AbstractMatrix, ncomponents_final::Integer; kwargs...) =
    gsvdnmf(truncating, X, ncomponents_final; kwargs...)

"""
    W_augmented, H_augmented, Λ = gsvdrecover([strategy,] X, W0, H0, kadd, f)

Augment the factorization `X ≈ W0*H0` with `kadd` additional components. This
is the augmentation step of [`gsvdnmf`](@ref), without the final NMF polish.

Candidate directions for the new rows of `H` are extracted from the
generalized SVD between `f` and the current factorization and ranked by
generalized singular value; `strategy` then assembles the non-negative
augmented factors.

Return `W_augmented` (`W0` with the new columns appended), `H_augmented`
(`H0` with the new rows appended), and `Λ`, the generalized singular values
of all `n1` candidate directions, in descending order.

# Arguments

- `strategy`: a callable
  `(X, W0, H0, Hadd, Λ, kadd) -> (W_augmented, H_augmented)` that assembles
  fully non-negative augmented factors from the candidate directions. `Hadd`
  contains *all* `n1` candidates as rows, aligned with `Λ`: `Λ[i]` is the
  generalized singular value of `Hadd[i, :]`, in descending order. `λ ≈ 1`
  marks a direction on which the data and the factorization agree; the
  interesting candidates are those with `λ` far from 1 (`λ ≫ 1`: present in
  the data but missing from the factorization, `Inf` meaning absent entirely;
  `λ ≪ 1`: present in the factorization but absent from the data). `kadd` is
  forwarded from the caller and each strategy defines its interpretation: the
  bundled [`GsvdInitialization.truncating`](@ref) (the default) and
  [`GsvdInitialization.joint_nnls`](@ref) add *exactly* `kadd` components,
  while [`GsvdInitialization.appending`](@ref) adds *at most* `kadd`. The
  number of components a strategy kept can be read off the width of the
  returned factors.
- `X`: non-negative data matrix. `X` need not be an `AbstractMatrix`: any
  object supporting the operations required by the chosen strategy (see
  [`truncating`](@ref) and [`joint_nnls`](@ref)) can be used, e.g. a lazy
  low-rank representation.
- `W0`: left factor of the existing factorization, of size `(m, n1)`.
- `H0`: right factor of the existing factorization, of size `(n1, p)`.
- `kadd`: the requested number of new components, forwarded to `strategy`
  (see above for its interpretation); must satisfy `1 ≤ kadd ≤ n1`.
- `f`: a singular value decomposition of `X` with at least `n1` components,
  e.g. from `LinearAlgebra.svd` or `TSVD.tsvd`. Any object whose factors `U`,
  `S`, `V` are indexable as `f[1]`, `f[2]`, `f[3]` works.

All array arguments, including the factors of `f`, must use 1-based indexing.

# Examples

```jldoctest
julia> using LinearAlgebra: svd

julia> X = Float64[1 0 0 1 0; 0 1 0 1 1; 0 0 1 0 1; 1 1 1 2 2];  # rank 3

julia> W0 = Float64[1 0; 0 1; 0 0; 1 1]; H0 = Float64[1 0 0 1 0; 0 1 0 1 1];  # rank-2 factorization of X

julia> W, H, Λ = gsvdrecover(X, W0, H0, 1, svd(X));

julia> size(W), size(H)
((4, 3), (3, 5))

julia> sum(abs2, X - W*H) < sum(abs2, X - W0*H0)  # the new component improves the fit
true
```
"""
function gsvdrecover(strategy, X, W0::AbstractMatrix, H0::AbstractMatrix, kadd::Integer, f)
    # `X` may be a non-array factored representation (see the docstring); only
    # arrays carry axes to validate.
    X isa AbstractArray && Base.require_one_based_indexing(X)
    Base.require_one_based_indexing(W0, H0)
    _, n = size(W0)
    kadd > 0 || throw(ArgumentError("kadd must be positive; got $kadd"))
    kadd <= n || throw(ArgumentError("the number of extra columns must be at most size(W0, 2); got kadd = $kadd, size(W0, 2) = $n"))
    size(first(f), 2) >= n || throw(ArgumentError("the supplied SVD has $(size(first(f), 2)) components but size(W0, 2) = $n are required"))
    U0, S0, V0 = f
    # An offset-axes SVD wider than `n` would make the `1:n` slices below
    # succeed on the wrong columns; reject it before slicing.
    Base.require_one_based_indexing(U0, S0, V0)
    U0, S0, V0 = U0[:, 1:n], S0[1:n], V0[:, 1:n]
    # The GSVD below treats `Diagonal(S0)` as nonsingular; a numerically
    # rank-deficient slice means `X` cannot support `n` components and the
    # candidate directions degenerate.
    S0[n] > sqrt(eps(float(real(eltype(S0))))) * S0[1] || throw(
        ArgumentError(
            "the supplied SVD is numerically rank deficient over its first $n components " *
                "(s[$n]/s[1] = $(S0[n] / S0[1])): the factorization is overcomplete relative to X, " *
                "and GSVD augmentation is not defined"
        )
    )
    Hadd, Λ = init_H(U0, S0, V0, W0, H0)
    ndeficient = count(isinf, Λ)
    kadd >= ndeficient || @warn "kadd ($kadd) is less than the rank deficiency of W0*H0 ($ndeficient)."
    W, H = strategy(X, W0, H0, Hadd, Λ, kadd)
    return W, H, Λ
end
gsvdrecover(X, W0::AbstractMatrix, H0::AbstractMatrix, kadd::Integer, f) =
    gsvdrecover(truncating, X, W0, H0, kadd, f)

"""
    truncating(X, W0, H0, Hadd, Λ, kadd) -> (W_augmented, H_augmented)

Default [`gsvdrecover`](@ref) strategy; adds **exactly** `kadd` components,
the leading (largest-`Λ`) candidate directions. Nonnegative least-squares
(NNLS) is used only for the rescaling weights `α` of the existing columns; the
new columns of `W` are computed by ordinary least squares and made
non-negative by an NNDSVD step, after which all columns are rebalanced.

This strategy requires only `*` and `sum(abs2, ·)` from `X`, so `X` may be a
lazy or factored low-rank representation rather than a materialized matrix.
See [`joint_nnls`](@ref) for an alternative that solves for the new columns
and the rescaling jointly, at greater cost.

Return non-negative `(W_augmented, H_augmented)`.
"""
function truncating(X, W0::AbstractMatrix, H0::AbstractMatrix, Hadd::AbstractMatrix, Λ::AbstractVector, kadd::Integer)
    X isa AbstractArray && Base.require_one_based_indexing(X)
    Base.require_one_based_indexing(W0, H0, Hadd)
    Hadd = Hadd[1:kadd, :]
    Wadd, a = init_W(X, W0, H0, Hadd)
    Wadd_nn, Hadd_nn = nndsvd(X, kadd, initdata = (U = Wadd, S = ones(eltype(Wadd), kadd), V = Hadd'))
    W0_1, H0_1 = [a' .* W0 Wadd_nn], [H0; Hadd_nn]
    cs = Wcols_modification(X, W0_1, H0_1)
    W0_2, H0_2 = cs' .* W0_1, H0_1
    return abs.(W0_2), abs.(H0_2)
end

"""
    appending(X, W0, H0, Hadd, Λ, kadd) -> (W_augmented, H_augmented)
    appending(thresh; rtol = nothing) -> strategy

[`gsvdrecover`](@ref) strategy that holds the existing factorization fixed:
the returned factors contain `W0` and `H0` unmodified, with new components
appended. It adds **at most** `kadd` components, applying three rejection
criteria to the candidates (considered in descending-`Λ` order):

- `Λ > thresh`: a direction with `λ ≤ thresh` does not carry enough excess
  energy in the data, relative to what the factorization already explains, to
  justify a new component. `appending` itself uses `thresh = 1`;
  `appending(thresh)` returns a strategy with a stricter (or looser) value.
- nonzero fitted amplitude: a candidate whose refit amplitude is zero cannot
  reduce the residual.
- relative energy above `rtol`: `Λ` is a ratio, so a direction of negligible
  absolute energy can still have a huge `λ` (e.g. exactly-rank-deficient data
  under an overcomplete factorization). A kept candidate's fitted contribution
  must satisfy `‖β·w*h'‖ > rtol·‖X‖` (Frobenius). The default
  `rtol = √eps(eltype)` rejects numerical dust only.

The new columns of `W` are computed by ordinary least squares on the residual
`X - W0*H0` (i.e., with the existing-column weights pinned at 1) and made
non-negative together with the candidate directions by an NNDSVD step, after
which their amplitudes are refit by NNLS (existing columns again pinned at 1).

Use this strategy when the existing components must not be perturbed — for
example when they are externally constrained, already validated, or shared
with a fit over a larger domain. The bundled alternatives [`truncating`](@ref)
and [`joint_nnls`](@ref) instead rescale the existing columns to rebalance the
augmented factorization as a whole.

Like [`truncating`](@ref), this strategy requires only `*` and `sum(abs2, ·)`
from `X`, so `X` may be a lazy or factored low-rank representation.

Return non-negative `(W_augmented, H_augmented)`.
"""
appending(X, W0::AbstractMatrix, H0::AbstractMatrix, Hadd::AbstractMatrix, Λ::AbstractVector, kadd::Integer) =
    _appending(1, nothing, X, W0, H0, Hadd, Λ, kadd)
appending(thresh::Real; rtol::Union{Real, Nothing} = nothing) =
    (X, W0, H0, Hadd, Λ, kadd) -> _appending(thresh, rtol, X, W0, H0, Hadd, Λ, kadd)

function _appending(thresh::Real, rtol, X, W0::AbstractMatrix, H0::AbstractMatrix, Hadd::AbstractMatrix, Λ::AbstractVector, kadd::Integer)
    X isa AbstractArray && Base.require_one_based_indexing(X)
    Base.require_one_based_indexing(W0, H0, Hadd)
    # `Λ` is in descending order, so the candidates above threshold are the
    # leading ones.
    nsel = min(kadd, count(>(thresh), Λ))
    nsel == 0 && return W0, H0
    Hadd = Hadd[1:nsel, :]
    Wadd, _ = init_W(X, W0, H0, Hadd; α = ones(eltype(W0), size(W0, 2)))
    Wadd_nn, Hadd_nn = nndsvd(X, nsel, initdata = (U = Wadd, S = ones(eltype(Wadd), nsel), V = Hadd'))
    Wadd_nn, Hadd_nn = abs.(Wadd_nn), abs.(Hadd_nn)
    # The NNDSVD step retains only one sign quadrant of each rank-1 term, which
    # perturbs its amplitude.  Refit the amplitudes β of just the new columns
    # (existing columns pinned at 1): min_{β ≥ 0} ‖X − W0*H0 − Σₖ βₖ wₖ hₖ‖².
    B = (Wadd_nn' * Wadd_nn) .* (Hadd_nn * Hadd_nn')
    a = diag(Wadd_nn' * X * Hadd_nn') - diag((Wadd_nn' * W0) * (H0 * Hadd_nn'))
    if isposdef(Symmetric(B))
        β = vec(nonneg_lsq(B, a; alg = :fnnls, gram = true))
        # Reject candidates that cannot reduce the residual (β = 0) or whose
        # fitted contribution is negligible at the scale of the data:
        # ‖βₖ wₖ hₖ'‖²_F = βₖ² B[k,k].
        rt = rtol === nothing ? sqrt(eps(float(real(eltype(B))))) : rtol
        floor2 = rt^2 * sum(abs2, X)
        sel = findall(k -> β[k]^2 * B[k, k] > floor2, eachindex(β))
        Wadd_nn = β[sel]' .* Wadd_nn[:, sel]
        Hadd_nn = Hadd_nn[sel, :]
    else
        # Degenerate Gram matrix (e.g. a zero or duplicated candidate column):
        # keep the NNDSVD amplitudes rather than risk a singular solve.
        sum(abs2, B) <= 1.0e-12 || @warn "B is not positive definite; keeping NNDSVD amplitudes for the appended components." maxlog = 1
    end
    return [W0 Wadd_nn], [H0; Hadd_nn]
end

"""
    joint_nnls(X, W0, H0, Hadd, Λ, kadd) -> (W_augmented, H_augmented)

Alternative [`gsvdrecover`](@ref) strategy that adds **exactly** `kadd`
components (the leading, largest-`Λ` candidates), solving for the new columns
of `W` and the rescaling `α` of the existing columns jointly, as a single
nonnegative least-squares (NNLS) problem. `Hadd` is first projected onto the
non-negative orthant, keeping whichever sign of each candidate direction
better matches the non-negative part of the residual `X - W0*H0`.

The joint NNLS problem has one unknown for every entry of the new columns of
`W` plus one rescaling weight per existing column, so this strategy is more
expensive than the default [`truncating`](@ref), especially when `X` has many
rows. Beyond the `*` and `sum(abs2, ·)` that `truncating` needs from `X`, it
also requires `X - W0*H0` and `eltype(X)`.

Return non-negative `(W_augmented, H_augmented)`.
"""
function joint_nnls(X, W0::AbstractMatrix, H0::AbstractMatrix, Hadd::AbstractMatrix, Λ::AbstractVector, kadd::Integer)
    X isa AbstractArray && Base.require_one_based_indexing(X)
    Base.require_one_based_indexing(W0, H0, Hadd)
    Hadd_nn = truncatepos(Hadd[1:kadd, :]', X, W0, H0)'
    Wadd, a = init_W_joint_nnls(X, W0, H0, Hadd_nn)
    W0_1, H0_1 = [a' .* W0 Wadd], [H0; Hadd_nn]
    return abs.(W0_1), abs.(H0_1)
end

function init_H(U0::AbstractMatrix, S0::AbstractVector, V0::AbstractMatrix, W0::AbstractMatrix, H0::AbstractMatrix)
    _, _, Q, D1, D2, R = svd(Matrix(Diagonal(S0)), (U0' * W0) * (H0 * V0))
    r0 = size(U0, 2)
    k = findfirst(x -> x != 0, D2[1, :])
    k = (k === nothing) ? r0 : k - 1
    F = (diag(D1[(k + 1):r0, (k + 1):r0]) ./ diag(D2[1:(r0 - k), (k + 1):r0])) .^ 2
    Λ = vcat(fill(Inf, k), F)
    H_index = sortperm(Λ, rev = true)
    # Columns of inv(R*Q') = Q*inv(R) in descending-Λ order, via a triangular
    # backsolve: the GSVD's Q is orthogonal and R is square upper triangular
    # (Diagonal(S0) is nonsingular, so k+l = r0).
    E = zeros(eltype(R), r0, r0)
    for (j, idx) in enumerate(H_index)
        E[idx, j] = 1
    end
    Hadd = Q * (UpperTriangular(R) \ E)
    Hadd_1 = V0 * Hadd
    # The outputs are aligned: both follow descending-Λ order, so `Λ[i]` is
    # the generalized singular value of `Hadd[i, :]`.
    return Hadd_1', Λ[H_index]
end

function init_W_joint_nnls(X, W0::AbstractMatrix{T}, H0::AbstractMatrix{T}, Hadd::AbstractMatrix{T}) where {T}
    m = size(X, 1)
    kadd = size(Hadd, 1)
    G = gram_sp_C(W0, H0, Hadd)[1]
    b = gram_b(X, W0, H0, Hadd)
    θ = nonneg_lsq(G, b; alg = :fnnls, gram = true)
    Wadd = reshape(θ[1:(m * kadd)], m, kadd)
    α = θ[(m * kadd + 1):end]
    return Wadd, α
end

function gram_sp_C(W0, H0, Hadd)
    m, r0 = size(W0)
    k = size(Hadd, 1)
    mk = m * k
    W0W0, H0H0 = W0' * W0, H0 * H0'
    P = Hadd * H0'
    HH = Hadd * Hadd'
    G22 = sparse(W0W0 .* H0H0)
    G12 = zeros(eltype(W0W0), mk, r0)
    for j in 1:r0
        G12[:, j] .= vec(W0[:, j] * P[:, j]')
    end
    G12 = sparse(G12)
    G11 = kronecker(HH, sparse(I, m, m))
    G = [G11 G12; G12' G22]
    return G, G11, G12, G22
end

function gram_b(X, W0, H0, Hadd)
    b = vcat(vec(X * Hadd'), diag(W0' * X * H0'))
    return b
end

function init_W(X, W0::AbstractMatrix{T}, H0::AbstractMatrix{T}, Hadd::AbstractMatrix{T}; α = nothing) where {T}
    A, b, _, cholHH, H0Hadd, XHaddt = obj_para(X, W0, H0, Hadd)
    if α === nothing
        if isposdef(A)
            α = nonneg_lsq(A, -b; alg = :fnnls, gram = true)
        else
            # A is not positive definite: the QP min_{α≥0} α'Aα + 2b'α has no
            # unique bounded minimum, so fnnls is not meaningful.  Fall back to
            # α = 1 (keep existing components at their current scale).
            sum(abs2, A) <= 1.0e-12 || @warn "A is not positive definite." maxlog = 1
            α = ones(T, size(A, 1))
        end
    end
    Wadd = (XHaddt - W0 * Diagonal(α[:]) * H0Hadd) / cholHH
    return Wadd, abs.(α)
end

function obj_para(X, W0::AbstractMatrix{T}, H0::AbstractMatrix{T}, Hadd::AbstractMatrix{T}) where {T}
    XHaddt = X * Hadd'
    H0Hadd = H0 * Hadd'
    HH = Hadd * Hadd'
    W0W0 = W0' * W0
    H0H0 = H0 * H0'
    cholHH = cholesky(Symmetric(HH))
    A = W0W0 .* (H0H0 - H0Hadd * (cholHH \ H0Hadd'))
    W0tXH0t = W0' * X * H0'
    W0XHaddt = W0' * XHaddt
    b = diag(H0Hadd * (cholHH \ W0XHaddt') - W0tXH0t)
    C = sum(abs2, X) - tr(cholHH \ (XHaddt' * XHaddt))
    return Symmetric(A), b, C, cholHH, H0Hadd, XHaddt
end

function Wcols_modification(X, W::AbstractMatrix{T}, H::AbstractMatrix{T}) where {T}
    WW, HH = W' * W, H * H'
    WtXHt = W' * X * H'
    a = diag(WtXHt)
    B = WW .* HH
    β = nonneg_lsq(B, a; alg = :fnnls, gram = true)
    return β[:]
end

function truncatepos(Y, X, W, H)
    ΔX = max.(zero(eltype(X)), X - W * H)
    Yout = similar(Y)
    for j in axes(Y, 2)
        y = view(Y, :, j)
        yp = max.(y, zero(eltype(y)))
        ym = max.(-y, zero(eltype(y)))
        if sum(ΔX * yp) >= sum(ΔX * ym)
            Yout[:, j] = yp
        else
            Yout[:, j] = ym
        end
    end
    return Yout
end


end
