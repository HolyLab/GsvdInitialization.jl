function generate_ground_truth()
    m, n, nfeatures = 150, 120, 10
    feature_sigmas = 2*ones(nfeatures)
    feature_centers = [i*10+10 for i in 1:nfeatures]
    feature_intensity = ones(nfeatures)
    W = W_init_gauss(m, feature_centers, feature_sigmas, feature_intensity)
    H = zeros(nfeatures, n)
    for r in 1:nfeatures
        h_start = r*10+7
        h_length = 10
        h_end = h_start+h_length-1
        H[r, h_start:h_end] .+= ones(1, h_length)'
    end
    return W, H
end

function W_init_gauss(n, centers, sigmas, intensity)
    W = []
    nc, ns, ni = length(centers), length(sigmas), length(intensity)
    nc == ns && ns == ni||throw(ArgumentError("centers, sigmas and intensity should have same length"))
    for i in 1:nc
        w = zeros(n)
        gauss_template = gaussiantemplate(sigmas[i])
        δ = Int(round((length(gauss_template)-1)/2))
        template_start = centers[i]-δ
        template_end = template_start + length(gauss_template)-1
        w[template_start:template_end] .+= intensity[i]*gauss_template
        push!(W, w)
    end
    return hcat(W...) 
end

function gaussiantemplate(T::Type, r::Real)
    len = round(Int, 8*r+1)
    w = (len - 1) ÷ 2
    template = Array{T}(undef, len)
    R2 = 2*r*r
    for x = -w:w
        template[x+w+1] = exp(-(x*x)/R2)
    end
    return template
end
gaussiantemplate(r::Real) = gaussiantemplate(Float64, r)

