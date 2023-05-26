module GsvdInitialization

using LinearAlgebra, NMF
using JuMP, Ipopt

export gsvdinit,
       init_H,
       init_W,
       Wcols_modification

function gsvdinit(X::AbstractArray, W0::AbstractArray, H0::AbstractArray, kadd::Int; initdata = nothing)
    if kadd == 0
        return W0, H0
    else
        m, n = size(W0)
        kadd <= n || throw(ArgumentError("# of extra columns must less than 1st NMF components"))
        U, S, V = initdata === nothing ? svd(X) : (initdata.U, initdata.S, initdata.V)
        U0, S0, V0 = U[:,1:n], S[1:n], V[:,1:n]
        Hadd = init_H(U0, S0, V0, W0, H0, kadd)
        Wadd, a = init_W(X, W0, H0, Hadd)
        Wadd_nn, Hadd_nn = NMF.nndsvd(X, kadd, initdata = (U = Wadd, S = ones(kadd), V = Hadd'))
        W0_1, H0_1 = [repeat(a', m, 1).*W0 Wadd_nn], [H0; Hadd_nn]
        cs = Wcols_modification(X, W0_1, H0_1)
        W0_2, H0_2 = repeat(cs', m, 1).*W0_1, H0_1
        return abs.(W0_2), abs.(H0_2)
    end
end

function init_H(U0::AbstractArray, S0::AbstractArray, V0::AbstractArray, W0::AbstractArray, H0::AbstractArray, kadd::Int)
    _, _, Q, D1, D2, R = svd(Matrix(Diagonal(S0)), (U0'*W0)*(H0*V0));
    # QtR = Q'*R
    inv_RQt = inv(R*Q')
    F = (diag(D1)./diag(D2)).^2
    if kadd < size(U0, 2)
        k0 = kadd
        H_index = []
        while k0 >= 1
            j = findmax(F)[2]
            F[j] = -1
            push!(H_index, j) 
            k0 -= 1   
        end
        Hadd = inv_RQt[:,H_index]
    else
        Hadd = inv_RQt
    end
    Hadd_1 = V0*Hadd
    return Hadd_1'
end

function init_W(X::AbstractArray{T}, W0::AbstractArray{T}, H0::AbstractArray{T}, Hadd::AbstractArray{T}; α = nothing) where T
    R = size(W0, 2)
    A, b, _, invHH, H0Hadd, XHaddt = obj_para(X, W0, H0, Hadd)
    if α === nothing 
        model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
        @variable(model, a[1:R] >= 1e-12, start = 1)
        @objective(model, Min, a'*A*a+2*b'*a)
        optimize!(model)
        α = JuMP.value.(a)
    end
    Wadd = XHaddt*invHH-W0*Diagonal(α)*H0Hadd*invHH
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
    return A, b, C, invHH, H0Hadd, XHaddt
end

function Wcols_modification(X::AbstractArray{T}, W::AbstractArray{T}, H::AbstractArray{T}) where T
    n = size(W, 2)
    a = Array{T}(undef, n)
    B = Array{T}(undef, n, n)
    WW, HH = W'*W, H*H' 
    WtXHt = W'*X*H'
    a = diag(WtXHt)
    B = WW.*HH
    model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
    @variable(model, b[1:n] >= 1e-12, start = 1)
    @objective(model, Min, b'*B*b-2*a'*b)
    optimize!(model)
    β = JuMP.value.(b)
    return β
end

end
