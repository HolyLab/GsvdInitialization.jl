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
    W0_2, H0_2 = repeat(cs', m, 1).*W0_1, H0_1
    return W0_2, H0_2, cs, W0_1, H0_1, a, Wadd, Hadd
end

function init_H(U0::AbstractArray, S0::AbstractArray, V0::AbstractArray, W0::AbstractArray, H0::AbstractArray, kadd::Int)
    _, _, Q, D1, D2, R = svd(Matrix(Diagonal(S0)), (U0'*W0)*(H0*V0));
    QtR = Q'*R
    F = (diag(D1)./diag(D2)).^2
    if kadd < size(U0, 2)
        k0 = kadd
        Hadd_vec = []
        while k0 >= 1
            j = findmax(F)[2]
            F[j] = -1
            # h = [Q[:,i]'*R[:,j] for i in axes(Q, 2)]
            # h = [QtR[i,j] for i in axes(Q, 2)]
            # h = copy(QtR[:,j])
            # push!(Hadd_vec, h) 
            push!(Hadd_vec, j) 
            k0 -= 1   
        end
        # Hadd = hcat(Hadd_vec...)
        Hadd = QtR[:,h]
    else
        Hadd = QtR
    end
    Hadd_1 = V0*Hadd
    return Hadd_1'
end

function init_W(X::AbstractArray, W0::AbstractArray, H0::AbstractArray, Hadd::AbstractArray; α = nothing)
    m, R = size(W0)
    K = size(Hadd, 1)
    A, b, _, invHH, H0Hadd, XHaddt = obj_para(X, W0, H0, Hadd)
    if α === nothing 
        model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
        @variable(model, a[1:R] >= 1e-12, start = 1)
        @objective(model, Min, a'*A*a+2*b'*a)
        optimize!(model)
        α = JuMP.value.(a)
    end
    # Wadd = zeros(m, K)
    # Wadd = XHaddt*invHH
    Wadd = XHaddt*invHH-W0*Diagonal(α)*H0Hadd*invHH
    # for j in 1:K
    #     for k1 in 1:K
    #         # Wadd[:,j] += invHH[j,k1]*γ[k1]
    #         for k2 in 1:R
    #             # Wadd[:,j] -= invHH[j,k1]*α[k2]*Hadd[k1,:]'*H0[k2,:]*W0[:,k2]
    #             aaa = invHH[j,k1]*α[k2]*H0Hadd[k2,k1]
        
    #             for i in 1:m 
    #                 Wadd[i,j] -= aaa*W0[i,k2]
    #             end
    #         end
    #     end
    # end
    return Wadd, abs.(α)
end

function obj_para(X::AbstractArray{T}, W0::AbstractArray, H0::AbstractArray, Hadd::AbstractArray) where T
    # R = size(W0, 2)
    # K = size(Hadd, 1)
    XHaddt = X*Hadd'
    # γ = [XHaddt[:,j] for j in 1:K]
    # HH = zeros(K, K)
    H0Hadd = H0*Hadd'
    # for k in 1:K, k1 in k:K
    #     HH[k, k1] = Hadd[k,:]'*Hadd[k1,:]
    #     HH[k1, k] = HH[k, k1]
    # end
    HH = Hadd*Hadd'
    W0W0 = W0'*W0
    H0H0 = H0*H0'
    invHH = inv(HH)
    # A = zeros(R, R)
    # A = W0W0.*H0H0
    A = W0W0.*(H0H0-H0Hadd*invHH*H0Hadd')
    # for i in 1:R, j in i:R
    #     # A[i,j] = (W0[:,i]'*W0[:,j])*(H0[i,:]'*H0[j,:])
    #     # A[i,j] = W0W0[i,j]*H0H0[i,j]
    #     for k1 in 1:K, k2 in 1:K
    #         # A[i,j] -= invHH[k1,k2]*(H0[i,:]'*Hadd[k1,:])*(W0[:,i]'*W0[:,j])*(H0[j,:]'*Hadd[k2,:])
    #         A[i,j] -= invHH[k1,k2]*H0Hadd[i,k1]*(W0W0[i,j])*H0Hadd[j,k2]
    #     end
    #     A[j,i] = A[i,j]
    # end
    # @show A
    W0tXH0t = W0'*X*H0'
    # b = -diag(W0tXH0t)
    W0XHaddt = W0'*XHaddt
    # b += diag(H0Hadd*invHH*W0XHaddt')

    b = diag(H0Hadd*invHH*W0XHaddt'-W0tXH0t)
    # for i in 1:R 
    #     # b[i] = -W0[:,i]'*(X*H0[i,:])
    #     for k1 in 1:K, k2 in 1:K
    #         # b[i] += invHH[k1,k2]*H0[i,:]'*Hadd[k1,:]*W0[:,i]'*γ[k2]    
    #         # b[i] += invHH[k1,k2]*H0Hadd[i,k1]*W0[:,i]'*γ[k2]   
    #         # b[i] += invHH[k1,k2]*H0Hadd[i,k1]*W0[:,i]'*XHaddt[:,k2]
    #         # b[i] += invHH[k1,k2]*H0Hadd[i,k1]*W0XHaddt[i,k2]              
    #     end
    # end
    C = sum(abs2, X)-sum(invHH.*(XHaddt'*XHaddt))
    # for k1 in 1:K, k2 in 1:K
    #     C -= invHH[k1,k2]*γ[k1]'*γ[k2]            
    # end
    return A, b, C, invHH, H0Hadd, XHaddt
end

function Wcols_modification(X::AbstractArray{T}, W::AbstractArray, H::AbstractArray) where T
    n = size(W, 2)
    a = Array{T}(undef, n)
    B = Array{T}(undef, n, n)
    WW, HH = W'*W, H*H' 
    WtXHt = W'*X*H'
    # for q in 1:n
    #     a[q] = WtXH[q,q]
    #     for p in q:n
    #         # B[p,q] = (W[:,p]'*W[:,q])*(H[p,:]'*H[q,:])
    #         B[p,q] = WW[p,q]*HH[p,q]
    #         B[q,p] = B[p,q]
    #     end
    # end
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


# function init_W(X::AbstractArray, W0::AbstractArray, H0::AbstractArray, Hadd::AbstractArray)
#     m, r = size(W0)
#     A, b, _, P, _, _, _, γ, inv_Π, _ = obj_para(X, W0, H0, Hadd)
#     model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
#     @variable(model, a[1:r] >= 1e-12, start = 1)
#     @objective(model, Min, a'*A*a+2*b'*a)
#     optimize!(model)
#     α = JuMP.value.(a)
#     Wadd = reshape(inv_Π*(γ-P'*α), m, size(Hadd, 1))
#     return Wadd, abs.(α)
# end

# function obj_para1(X::AbstractArray{T}, W0::AbstractArray, H0::AbstractArray, Hadd::AbstractArray) where T
#     m, R = size(W0)
#     K = size(Hadd, 1)
#     ξ = Array{T}(undef, R)
#     Θ = Array{T}(undef, R, R)
#     for q in 1:R
#         ξ[q] = W0[:,q]'*X*H0[q,:]
#         for p in q:R
#             Θ[p,q] = (W0[:,p]'*W0[:,q])*(H0[p,:]'*H0[q,:])
#             Θ[q,p] = Θ[p,q]
#         end
#     end
#     Φ = sum(abs2, X)
#     γ = [X*Hadd'...]
#     P0 = []
#     for i in 1:K
#         v = []
#         for r in 1:R 
#             push!(v, (H0[r,:]'*Hadd[i,:])*W0[:,r]')
#         end
#         push!(P0, vcat(v...))
#     end
#     P = hcat(P0...)
#     inv_Π = zeros(K*m, K*m)
#     HH = zeros(K, K)
#      for k in 1:K, k1 in k:K
#         HH[k, k1] = Hadd[k,:]'*Hadd[k1,:]
#         HH[k1, k] = HH[k, k1]
#     end
#     inv_HH = inv(HH)   
#     for k in 0:K-1, k1 in k:K-1
#         inv_Π[k*m+1:(k+1)*m, k1*m+1:(k1+1)*m] = inv_HH[k+1, k1+1]*Matrix{Float64}(I, m, m)
#         inv_Π[k1*m+1:(k1+1)*m, k*m+1:(k+1)*m] = inv_Π[k*m+1:(k+1)*m, k1*m+1:(k1+1)*m]
#     end
#     A, b, C = Θ-P*(inv_Π*P'), P*(inv_Π*γ)-ξ, Φ-γ'*(inv_Π*γ)
#     return A, b, C, P, Θ, ξ, Φ, γ, inv_Π, HH
# end
