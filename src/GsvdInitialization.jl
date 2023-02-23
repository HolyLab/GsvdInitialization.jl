module GsvdInitialization

using LinearAlgebra, NMF
using JuMP, Ipopt

export gsvdinit,
       init_H,
       init_W,
       Wcols_modification

function gsvdinit(X::AbstractArray, W0::AbstractArray, H0::AbstractArray, kadd::Int; initdata = nothing)
    m, n = size(W0)
    kadd <= n || throw(ArgumentError("# of extra columns must less than 1st NMF components"))
    U, S, V = initdata === nothing ? svd(X) : (initdata.U, initdata.S, initdata.V)
    U0, S0, V0 = U[:,1:n], S[1:n], V[:,1:n]
    Hadd = init_H(U0, S0, V0, W0, H0, kadd)
    Wadd, a = init_W(X, W0, H0, Hadd)
    Wadd_nn, Hadd_nn = NMF.nndsvd(X, kadd, initdata = (U = Wadd, S = ones(kadd), V = Hadd'))
    W0_1, H0_1 = [repeat(a', m, 1).*W0 Wadd_nn], [H0; Hadd_nn]
    cs = Wcols_modification(X, W0_1, H0_1)
    W0_2, H0_2 = repeat(cs', m, 1).* W0_1, H0_1
    return W0_2, H0_2, cs, W0_1, H0_1, a, Wadd, Hadd
end

function init_H(U0::AbstractArray, S0::AbstractArray, V0::AbstractArray, W0::AbstractArray, H0::AbstractArray, kadd::Int)
    _, _, Q, D1, D2, R = svd(Matrix(Diagonal(S0)), (U0'*W0)*(H0*V0));
    F = (diag(D1)./diag(D2)).^2
    if kadd < size(U0, 2)
        k0 = kadd
        Hadd_vec = []
        while k0 >= 1
            j = findmax(F)[2]
            F[j] = -1
            h = [Q[:,i]'*R[:,j] for i in axes(Q, 2)]
            push!(Hadd_vec, h) 
            k0 -= 1   
        end
        Hadd = hcat(Hadd_vec...)
    else
        Hadd = Q'*R
    end
    Hadd_1 = V0*Hadd
    return Hadd_1'
end

function init_W(X::AbstractArray, W0::AbstractArray, H0::AbstractArray, Hadd::AbstractArray)
    m, r = size(W0)
    A, b, _, P, _, _, _, γ, Π = obj_para(X, W0, H0, Hadd)
    model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
    @variable(model, a[1:r] >= 1e-12, start = 1)
    @objective(model, Min, a'*A*a+2*b'*a)
    optimize!(model)
    α = JuMP.value.(a)
    Wadd = reshape(Π\(γ-P'*α), m, size(Hadd, 1))
    return Wadd, abs.(α)
end

function obj_para(X::AbstractArray{T}, W0::AbstractArray, H0::AbstractArray, Hadd::AbstractArray) where T
    m, R = size(W0)
    K = size(Hadd, 1)
    ξ = Array{T}(undef, R)
    Θ = Array{T}(undef, R, R)
    for q in 1:R
        ξ[q] = W0[:,q]'*X*H0[q,:]
        for p in q:R
            Θ[p,q] = (W0[:,p]'*W0[:,q])*(H0[p,:]'*H0[q,:])
            Θ[q,p] = Θ[p,q]
        end
    end
    Φ = sum(abs2, X)
    γ = [X*Hadd'...]
    P0 = []
    for i in 1:K
        v = []
        for r in 1:R 
            push!(v, (H0[r,:]'*Hadd[i,:])*W0[:,r]')
        end
        push!(P0, vcat(v...))
    end
    P = hcat(P0...)
    Π = zeros(K*m, K*m)
    for k in 0:K-1, k1 in k:K-1
        Π[k*m+1:(k+1)*m, k1*m+1:(k1+1)*m] = Matrix(Diagonal(Hadd[k+1,:]'*Hadd[k1+1,:]*ones(m)))[:,:]
        Π[k1*m+1:(k1+1)*m, k*m+1:(k+1)*m] = Π[k*m+1:(k+1)*m, k1*m+1:(k1+1)*m]
    end
    A, b, C = Θ-P*(Π\P'), P*(Π\γ)-ξ, Φ-γ'*(Π\γ)
    return A, b, C, P, Θ, ξ, Φ, γ, Π
end

function Wcols_modification(X::AbstractArray{T}, W::AbstractArray, H::AbstractArray) where T
    n = size(W, 2)
    a = Array{T}(undef, n)
    B = Array{T}(undef, n, n)
    for q in 1:n
        a[q] = W[:,q]'*X*H[q,:]
        for p in q:n
            B[p,q] = (W[:,p]'*W[:,q])*(H[p,:]'*H[q,:])
            B[q,p] = B[p,q]
        end
    end
    model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
    @variable(model, b[1:n] >= 1e-12, start = 1)
    @objective(model, Min, b'*B*b-2*a'*b)
    optimize!(model)
    β = JuMP.value.(b)
    return β
end

end

# function objective_W(X::AbstractArray{T}, α::AbstractArray, W0::AbstractArray, H0::AbstractArray, Hadd::AbstractArray) where T
#     A, b, C, P, Θ, ξ, Φ, γ, Π = obj_para(X, W0, H0, Hadd)
#     E = α'*A*α+2*b'*α+C
#     return A, b, C, P, Θ, ξ, Φ, γ, Π, E
# end

# function objective_W_2(X::AbstractArray{T}, α::AbstractArray, W0::AbstractArray, H0::AbstractArray, Wadd::AbstractArray, Hadd::AbstractArray) where T
#     l = [Wadd...]
#     _, _, _, P, Θ, ξ, Φ, γ = obj_para(X, W0, H0, Hadd)
#     E = α'*Θ*α-2*ξ'*α+Φ-2*γ'*l+2*α'*P*l+l'*l
#     return E, P, Θ, ξ, Φ, γ
# end

# function gsvdinit(U0::AbstractArray, S0::AbstractArray, V0::AbstractArray, X::AbstractArray, W0::AbstractArray, H0::AbstractArray, kadd::Int)
#     m = size(W0, 1)
#     Hadd = init_H(U0, S0, V0, W0, H0, kadd)
#     Wadd, a = init_W(X, W0, H0, Hadd)
#     Wadd_nn, Hadd_nn = nndsvd(X, kadd, initdata = (U = Wadd, S = ones(kadd), V = Hadd'))
#     W0_1, H0_1 = [repeat(a', m, 1).*W0 Wadd_nn], [H0; Hadd_nn]
#     cs = Wcols_modification(X, W0_1, H0_1)
#     W0_2, H0_2 = repeat(cs', m, 1).* W0_1, H0_1
#     return W0_2, H0_2, cs, W0_1, H0_1, a, Wadd, Hadd
# end