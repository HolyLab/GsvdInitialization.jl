module GsvdInitialization

using LinearAlgebra, NMF, TSVD
using NonNegLeastSquares

export gsvdnmf,
       gsvdrecover

"""
    W, H = **gsvdnmf**(X::AbstractMatrix, W::AbstractMatrix, H::AbstractMatrix, f; 
                    n2 = size(first(f), 2), 
                    tol_nmf=1e-4, 
                    kwargs...)

    This funtion augments components for ``W`` and ``H``, and subsequently polishs new ``W`` and ``H`` by NMF.

    Arguments:

    ``X``: non-nagetive 2D data matrix

    ``W``: initialization of initial NMF

    ``H``: initialization of initial NMF

    ``n2``: the number of components in augmented matrix

    ``f``: SVD (or Truncated SVD) of ``X``, ``f`` needs to be explicitly writen in ``Tuple`` form.

    Keyword arguments 

    ``tol_nmf``: the tolerance of  NMF polishing step, default: 1e-4

    Other keyword arguments are passed to ``NMF.nnmf``.
"""
function gsvdnmf(X::AbstractMatrix, W::AbstractMatrix, H::AbstractMatrix, f; 
                 n2 = size(first(f), 2), 
                 tol_nmf=1e-4, 
                 kwargs...)
    n1 = size(W, 2)
    kadd = n2 - n1
    kadd >= 0 || throw(ArgumentError("The number of components to add must be non-negative"))
    kadd <= n1 || throw(ArgumentError("The number of components to add must be less than initial number of components"))
    size(first(f), 2) >= n1 || throw(ArgumentError("The supplied SVD does not have enough components"))
    if kadd == 0
        return W, H
    else
        W_recover, H_recover = gsvdrecover(X, copy(W), copy(H), kadd, f)
        result_recover = nnmf(X, n2; kwargs..., init=:custom, tol=tol_nmf, W0=W_recover, H0=H_recover)
        return result_recover.W, result_recover.H
    end
end
gsvdnmf(X::AbstractMatrix, W::AbstractMatrix, H::AbstractMatrix, n2::Int; kwargs...) = gsvdnmf(X, W, H, tsvd(X, n2); kwargs...)

"""
    W, H = **gsvdnmf**(X::AbstractMatrix, ncomponents::Pair{Int,Int}; tol_final=1e-4, tol_intermediate=1e-4, kwargs...)

    This function performs "GSVD-NMF" on 2D data matrix ``X``.

    Arguments:

    ``X``: non-nagetive 2D data matrix

    ``ncomponents::Pair{Int,Int}``: in the form of ``n1 => n2``, augments from ``n1`` components to ``n2``components, where ``n1`` is the number of components for initial NMF (under-complete NMF), and ``n2`` is the number of components for final NMF.

    Alternatively, ``ncomponents`` can be an integer denoting the number of components for final NMF. 
    In this case, ``gsvdnmf`` defaults to augment components on initial NMF solution by 1.

    Keyword arguments:

    ``tol_final``： The tolerence of final NMF, default:``10^{-4}``

    ``tol_intermediate``: The tolerence of initial NMF (under-complete NMF), default: tol_final

    Other keyword arguments are passed to ``NMF.nnmf``.
"""
function gsvdnmf(X::AbstractMatrix, ncomponents::Pair{Int,Int}; tol_final=1e-4, tol_intermediate=1e-4, kwargs...)
    n1, n2 = ncomponents
    f = tsvd(X, n2)
    W0, H0 = NMF.nndsvd(X, n1; initdata = (U = f[1], S = f[2], V = f[3]))
    result_initial_nmf = nnmf(X, n1; kwargs..., init=:custom, tol=tol_intermediate, W0=copy(W0), H0=copy(H0))
    W_initial_nmf, H_initial_nmf = result_initial_nmf.W, result_initial_nmf.H
    return gsvdnmf(X, W_initial_nmf, H_initial_nmf, f; kwargs..., n2=n2, tol_nmf=tol_final)
end
gsvdnmf(X::AbstractMatrix, ncomponents_final::Integer; kwargs...) = gsvdnmf(X, ncomponents_final-1 => ncomponents_final; kwargs...)

"""
    Wadd, Hadd, S = **gsvdrecover**(X, W0, H0, kadd, f)

    This funtion augments components for ``W`` and ``H`` without polishing NMF step.

    Outputs:

    ``Wadd``: augmented NMF solution

    ``Hadd``: augmented NMF solution

    ``S``: related generalized singular value

    Arguments:

    ``X``: non-nagetive 2D data matrix

    ``W0``: NMF solution

    ``H0``: NMF solution

    ``kadd``: number of new components

    ``f``: SVD (or Truncated SVD) of ``X``, ``f`` needs to be indexable.
"""
function gsvdrecover(X::AbstractArray, W0::AbstractArray, H0::AbstractArray, kadd::Int, f::Tuple)
    m, n = size(W0)
    kadd <= n || throw(ArgumentError("# of extra columns must less than 1st NMF components"))
    if kadd == 0
        return W0, H0, 0
    else
        U0, S0, V0 = f[1][:,1:n], f[2][1:n], f[3][:,1:n]
        Hadd, Λ = init_H(U0, S0, V0, W0, H0, kadd)
        Wadd, a = init_W(X, W0, H0, Hadd)
        Wadd_nn, Hadd_nn = NMF.nndsvd(X, kadd, initdata = (U = Wadd, S = ones(kadd), V = Hadd'))
        W0_1, H0_1 = [repeat(a', m, 1).*W0 Wadd_nn], [H0; Hadd_nn]
        cs = Wcols_modification(X, W0_1, H0_1)
        W0_2, H0_2 = repeat(cs', m, 1).*W0_1, H0_1
        return abs.(W0_2), abs.(H0_2), Λ
    end
end

function init_H(U0::AbstractArray, S0::AbstractArray, V0::AbstractArray, W0::AbstractArray, H0::AbstractArray, kadd::Int)
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
    return Hadd_1', Λ[H_index]
end

function init_W(X::AbstractArray{T}, W0::AbstractArray{T}, H0::AbstractArray{T}, Hadd::AbstractArray{T}; α = nothing) where T
    A, b, _, invHH, H0Hadd, XHaddt = obj_para(X, W0, H0, Hadd)
    (isposdef(A) || sum(abs2, A) <= 1e-12) || @warn "A is not positive definite."
    α = α === nothing ? nonneg_lsq(A, -b; alg=:fnnls, gram=true) : α
    Wadd = XHaddt*invHH-W0*Diagonal(α[:])*H0Hadd*invHH
    return Wadd, abs.(α)
end

function obj_para(X::AbstractArray{T}, W0::AbstractArray{T}, H0::AbstractArray{T}, Hadd::AbstractArray{T}) where T
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

function Wcols_modification(X::AbstractArray{T}, W::AbstractArray{T}, H::AbstractArray{T}) where T
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

end