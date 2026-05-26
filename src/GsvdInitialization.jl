module GsvdInitialization

using LinearAlgebra: Diagonal, I, Symmetric, diag, isposdef, svd
using NMF: NMF, nnmf, nndsvd
using TSVD: tsvd
using NonNegLeastSquares: nonneg_lsq
using Kronecker: kronecker
using SparseArrays: sparse

export gsvdnmf,
       gsvdrecover

@static if VERSION >= v"1.11"
    eval(Meta.parse("public truncating, joint_nnls"))
end

"""
    result, Λ = gsvdnmf([strategy,] X::AbstractMatrix, W::AbstractMatrix, H::AbstractMatrix, f;
                       n2 = size(first(f), 2),
                       tol_final=1e-4,
                       kwargs...)

Augment `W` and `H` to have `n2` components, subsequently polished by NMF.

Arguments:

- `strategy`: a callable `(X, W0, H0, Hadd) -> (W_augmented, H_augmented)` that
  produces fully-assembled non-negative augmented factors from the candidate
  directions `Hadd` ranked by [`init_H`](@ref). Defaults to
  [`GsvdInitialization.truncating`](@ref); [`GsvdInitialization.joint_nnls`](@ref) is
  the alternative (joint-NNLS) strategy.

- `X`: non-negative data matrix

- `W` and `H`: initial NMF factorization

- `n2`: the number of components in augmented factorization

- `f`: SVD (or Truncated SVD) of `X`

Keyword arguments:

- `tol_final`: the tolerance of the NMF polishing step, default: 1e-4

Other keyword arguments are passed to `NMF.nnmf`.

Returns the `NMF.NMFResult` from the polishing step (its `W` and `H` fields hold
the augmented factors) and `Λ`, the generalized singular values that ranked the
candidate augmentation directions.
"""
function gsvdnmf(strategy, X::AbstractMatrix, W::AbstractMatrix, H::AbstractMatrix, f;
                 n2 = size(first(f), 2),
                 tol_final = 1e-4,
                 alg = :cd,
                 truncmult = 1e-5,
                 kwargs...)
    n1 = size(W, 2)
    kadd = n2 - n1
    kadd > 0 || throw(ArgumentError("The number of components to add must be positive; got n2 = $n2, size(W, 2) = $n1"))
    kadd <= n1 || throw(ArgumentError("The number of components to add must be less than initial number of components"))
    size(first(f), 2) >= n1 || throw(ArgumentError("The supplied SVD does not have enough components"))
    W_recover, H_recover, Λ = gsvdrecover(strategy, X, copy(W), copy(H), kadd, f)
    if alg == :multmse
        W_recover, H_recover = max.(W_recover, truncmult), max.(H_recover, truncmult)
    end
    result_recover = nnmf(X, n2; kwargs..., init=:custom, tol=tol_final, W0=copy(W_recover), H0=copy(H_recover))
    return result_recover, Λ
end
gsvdnmf(X::AbstractMatrix, W::AbstractMatrix, H::AbstractMatrix, f; kwargs...) =
    gsvdnmf(truncating, X, W, H, f; kwargs...)
gsvdnmf(strategy, X::AbstractMatrix, W::AbstractMatrix, H::AbstractMatrix, n2::Integer; kwargs...) =
    gsvdnmf(strategy, X, W, H, tsvd(X, n2); kwargs...)
gsvdnmf(X::AbstractMatrix, W::AbstractMatrix, H::AbstractMatrix, n2::Integer; kwargs...) =
    gsvdnmf(truncating, X, W, H, n2; kwargs...)

"""
    result, Λ = gsvdnmf([strategy,] X::AbstractMatrix, ncomponents::Pair{Int,Int}; tol_final=1e-4, tol_intermediate=1e-4, kwargs...)

Perform "GSVD-NMF" on the data matrix `X`.

Arguments:

- `strategy`: see the four-argument [`gsvdnmf`](@ref) method. Defaults to
  [`GsvdInitialization.truncating`](@ref).

- `X`: non-negative data matrix

- `ncomponents`: in the form of `n1 => n2`, augments from `n1` components to `n2`components,
  where `n1` is the number of components for initial NMF (under-complete NMF), and `n2` is the number of
  components for final NMF.

Alternatively, `ncomponents` can be an integer denoting the number of components for final NMF.
In this case, `gsvdnmf` defaults to augment components on initial NMF solution by 1.

Keyword arguments:

- `tol_final`: The tolerence of final NMF, default:`10^{-4}`

- `tol_intermediate`: The tolerence of initial NMF (under-complete NMF), default: tol_final

Other keyword arguments are passed to `NMF.nnmf`.
"""
function gsvdnmf(strategy, X::AbstractMatrix, ncomponents::Pair{<:Integer,<:Integer}; tol_final=1e-4, tol_intermediate=tol_final, kwargs...)
    n1, n2 = ncomponents
    f = tsvd(X, n2)
    W0, H0 = nndsvd(X, n1; initdata = (U = f[1], S = f[2], V = f[3]))
    result_initial_nmf = nnmf(X, n1; kwargs..., init=:custom, tol=tol_intermediate, W0=copy(W0), H0=copy(H0))
    W_initial_nmf, H_initial_nmf = result_initial_nmf.W, result_initial_nmf.H
    return gsvdnmf(strategy, X, W_initial_nmf, H_initial_nmf, f; kwargs..., n2=n2, tol_final)
end
gsvdnmf(X::AbstractMatrix, ncomponents::Pair{<:Integer,<:Integer}; kwargs...) =
    gsvdnmf(truncating, X, ncomponents; kwargs...)
gsvdnmf(strategy, X::AbstractMatrix, ncomponents_final::Integer; kwargs...) =
    gsvdnmf(strategy, X, ncomponents_final-1 => ncomponents_final; kwargs...)
gsvdnmf(X::AbstractMatrix, ncomponents_final::Integer; kwargs...) =
    gsvdnmf(truncating, X, ncomponents_final; kwargs...)

"""
    W_augmented, H_augmented, Λ = gsvdrecover([strategy,] X, W0, H0, kadd, f)

Augment components for `W0` and `H0` without polishing by NMF.

`strategy` is a callable `(X, W0, H0, Hadd) -> (W_augmented, H_augmented)` that
produces fully-assembled non-negative augmented factors from the candidate
directions `Hadd` ranked by [`init_H`](@ref). Defaults to
[`GsvdInitialization.truncating`](@ref); [`GsvdInitialization.joint_nnls`](@ref) is the
alternative bundled strategy.

Outputs:

`W_augmented`, `H_augmented`: the full augmented NMF factors (with `kadd` extra
components appended to `W0`/`H0`)

`Λ`: generalized singular values used to rank the candidate augmentation directions

Arguments:

`X`: non-negative 2D data matrix

`W0`: NMF solution

`H0`: NMF solution

`kadd`: number of new components

`f`: SVD (or Truncated SVD) of `X`
"""
function gsvdrecover(strategy, X, W0::AbstractMatrix, H0::AbstractMatrix, kadd::Integer, f)
    _, n = size(W0)
    kadd > 0 || throw(ArgumentError("kadd must be positive; got $kadd"))
    kadd <= n || throw(ArgumentError("# of extra columns must less than 1st NMF components"))
    U0, S0, V0 = f
    U0, S0, V0 = U0[:,1:n], S0[1:n], V0[:,1:n]
    Hadd, Λ = init_H(U0, S0, V0, W0, H0, kadd)
    W, H = strategy(X, W0, H0, Hadd)
    return W, H, Λ
end
gsvdrecover(X, W0::AbstractMatrix, H0::AbstractMatrix, kadd::Integer, f) =
    gsvdrecover(truncating, X, W0, H0, kadd, f)

"""
    truncating(X, W0, H0, Hadd) -> (W_augmented, H_augmented)

Default `gsvdrecover` strategy. Restricts the use of nonnegative least-squares (NNLS)
to the component weights `α`, and uses least-squares followed by an NNDSVD step
to solve for the new columns of `W` (i.e., `Wadd`).

Returns non-negative `(W_augmented, H_augmented)`.
"""
function truncating(X, W0::AbstractMatrix, H0::AbstractMatrix, Hadd::AbstractMatrix)
    m = size(W0, 1)
    kadd = size(Hadd, 1)
    Wadd, a = init_W(X, W0, H0, Hadd)
    Wadd_nn, Hadd_nn = nndsvd(X, kadd, initdata = (U = Wadd, S = ones(kadd), V = Hadd'))
    W0_1, H0_1 = [repeat(a', m, 1).*W0 Wadd_nn], [H0; Hadd_nn]
    cs = Wcols_modification(X, W0_1, H0_1)
    W0_2, H0_2 = repeat(cs', m, 1).*W0_1, H0_1
    return abs.(W0_2), abs.(H0_2)
end

"""
    joint_nnls(X, W0, H0, Hadd) -> (W_augmented, H_augmented)

Alternative `gsvdrecover` strategy that jointly solves for the new columns of
`W` and the rescaling `α` of existing columns using nonnegative least-squares
(NNLS). `Hadd` is first projected onto the non-negative orthant.

Returns non-negative `(W_augmented, H_augmented)`.
"""
function joint_nnls(X, W0::AbstractMatrix, H0::AbstractMatrix, Hadd::AbstractMatrix)
    m = size(W0, 1)
    Hadd_nn = truncatepos(Hadd', X, W0, H0)'
    Wadd, a = init_W_joint_nnls(X, W0, H0, Hadd_nn)
    W0_1, H0_1 = [repeat(a', m, 1).*W0 Wadd], [H0; Hadd_nn]
    return abs.(W0_1), abs.(H0_1)
end

function init_H(U0::AbstractMatrix, S0::AbstractVector, V0::AbstractMatrix, W0::AbstractMatrix, H0::AbstractMatrix, kadd::Integer)
    _, _, Q, D1, D2, R = svd(Matrix(Diagonal(S0)), (U0'*W0)*(H0*V0));
    inv_RQt = inv(R*Q')
    r0 = size(U0, 2)
    k = findfirst(x->x!=0, D2[1,:])
    k = (k === nothing) ? r0 : k-1
    kadd >= k || @warn "kadd is less than rank deficiency of W0*H0."
    F = (diag(D1[k+1:r0, k+1:r0])./diag(D2[1:r0-k,k+1:r0])).^2
    Λ = vcat(fill(Inf, k), F)
    H_index = sortperm(Λ, rev = true)[1:kadd]
    Hadd = inv_RQt[:, H_index]
    Hadd_1 = V0*Hadd
    return Hadd_1', Λ
end

function init_W_joint_nnls(X::AbstractMatrix{T}, W0::AbstractMatrix{T}, H0::AbstractMatrix{T}, Hadd::AbstractMatrix{T}) where T
    m = size(X, 1)
    kadd = size(Hadd, 1)
    G = gram_sp_C(W0, H0, Hadd)[1]
    b = gram_b(X, W0, H0, Hadd)
    θ = nonneg_lsq(G, b; alg=:fnnls, gram=true)
    Wadd = reshape(θ[1:m*kadd], m, kadd)
    α = θ[m*kadd+1:end]
    return Wadd, α
end

function gram_sp_C(W0, H0, Hadd)
    m, r0 = size(W0)
    k = size(Hadd, 1)
    mk = m*k
    W0W0, H0H0 = W0'*W0, H0*H0'
    P = Hadd*H0'
    HH = Hadd*Hadd'
    G22 = sparse(W0W0.*H0H0)
    G12 = zeros(Float64, mk, r0)
    for j in 1:r0
        G12[:,j] .= vec(W0[:,j] * P[:,j]')
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

function init_W(X, W0::AbstractMatrix{T}, H0::AbstractMatrix{T}, Hadd::AbstractMatrix{T}; α = nothing) where T
    A, b, _, invHH, H0Hadd, XHaddt = obj_para(X, W0, H0, Hadd)
    if α === nothing
        if isposdef(A)
            α = nonneg_lsq(A, -b; alg=:fnnls, gram=true)
        else
            # A is not positive definite: the QP min_{α≥0} α'Aα + 2b'α has no
            # unique bounded minimum, so fnnls is not meaningful.  Fall back to
            # α = 1 (keep existing components at their current scale).
            sum(abs2, A) <= 1e-12 || @warn "A is not positive definite."
            α = ones(T, size(A, 1))
        end
    end
    Wadd = XHaddt*invHH-W0*Diagonal(α[:])*H0Hadd*invHH
    return Wadd, abs.(α)
end

function obj_para(X, W0::AbstractMatrix{T}, H0::AbstractMatrix{T}, Hadd::AbstractMatrix{T}) where T
    XHaddt = X*Hadd'
    H0Hadd = H0*Hadd'
    HH = Hadd*Hadd'
    W0W0 = W0'*W0
    H0H0 = H0*H0'
    invHH = inv(HH)
    A = W0W0.*(H0H0-H0Hadd*invHH*H0Hadd')
    W0tXH0t = W0'*X*H0'
    W0XHaddt = W0'*XHaddt
    b = diag(H0Hadd*invHH*W0XHaddt'-W0tXH0t)
    C = sum(abs2, X)-sum(invHH.*(XHaddt'*XHaddt))
    return Symmetric(A), b, C, invHH, H0Hadd, XHaddt
end

function Wcols_modification(X, W::AbstractMatrix{T}, H::AbstractMatrix{T}) where T
    n = size(W, 2)
    a = Array{T}(undef, n)
    B = Array{T}(undef, n, n)
    WW, HH = W'*W, H*H'
    WtXHt = W'*X*H'
    a = diag(WtXHt)
    B = WW.*HH
    β = nonneg_lsq(B, a; alg=:fnnls, gram=true)
    return β[:]
end

function truncatepos(Y, X, W, H)
    ΔX = max.(zero(eltype(X)), X - W*H)
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