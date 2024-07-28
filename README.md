# GsvdInitialization

[![CI](https://github.com/HolyLab/GsvdInitialization.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/HolyLab/GsvdInitialization.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/HolyLab/GsvdInitialization.jl/graph/badge.svg?token=LxqRCsZIvn)](https://codecov.io/gh/HolyLab/GsvdInitialization.jl)

This package includes the code of the paper 'GSVD-NMF: Recovering Missing Features in
Non-negative Matrix Factorization`. 
It is used to recover Non-negative matrix factorization(NMF) components from low-dimensional space to higher dimensional space by exploiting the generalized singular value decomposition (GSVD) between existing NMF results and the SVD of X.
This method allows the incremental expansion of the number of components, which can be convenient and effective for interactive analysis of large-scale data.

See also [NMFMerge](https://github.com/HolyLab/NMFMerge.jl) for the converse operation. Together, the two result in a substantial improvement in the quality and consistency of NMF factorization.
---------------------------

Demo:

To run this demo, NMF.jl and LinearAlgebra.jl are also required.

Install and load packages
```julia
julia>] add GsvdInitialization;
julia> using GsvdInitialization, NMF, LinearAlgebra;
```

Generating grouth truth with 10 features.

```julia
julia> include("demo/generate_ground_truth.jl")
julia> W_GT, H_GT = generate_ground_truth();
julia> X = W_GT*H_GT;
```

<img src="demo/GroundTruth.png" alt="Sample Figure" width="400"/>

Running standard NMF(HALS) using NNDSVD as initialization on X. Here, we're taking a couple of precautions to try to ensure the best possible result from NMF:
- we disable premature convergence by setting `maxiter` to something that is practically infinite
- we use the full `svd`, rather than `rsvd`, for initializing NNDSVD, as `svd` gives higher-quality results than `rsvd`
Despite these precautions, we'll see that the NMF result leaves much to be desired:

```julia
julia> result_hals = nnmf(X, 10; init=:nndsvd, alg = :cd, tol = 1e-4, maxiter=10^12, initdata = svd(X));
julia> sum(abs2, X-result_hals.W*result_hals.H)/sum(abs2, X)
0.0999994991270576
```
The result is given by

<img src="demo/ResultHals.png" alt="Sample Figure" width="400"/>

This factorization is not perfect as two components are same and two features share one component.
Then, running GSVD-NMF on X (also using NNSVD as initialization).

```julia
Wgsvd, Hgsvd = gsvdnmf(X, 9=>10; alg = :cd, tol_final = 1e-4, tol_intermediate = 1e-2, maxiter = 10^12);
julia> sum(abs2, X-Wgsvd*Hgsvd)/sum(abs2, X)
1.2322603074132593e-10
```
Gsvd-NMF factorizes the gound truth well based on the comparison between relative fitting errors and figures.

<img src="demo/ResultGsvdNMF.png" alt="Sample Figure" width="400"/>


---------------------------

## Functions

W, H = **gsvdnmf**(X::AbstractMatrix, ncomponents::Pair{Int,Int}; kwargs...)

This function performs "GSVD-NMF" on 2D data matrix ``X``.

Arguments:

``ncomponents::Pair{Int,Int}``: in the form of ``n1 => n2``, augments from ``n1`` components to ``n2``components, where ``n1`` is the number of components for initial NMF (under-complete NMF), and ``n2`` is the number of components for final NMF.

Alternatively, ``ncomponents`` can be an integer denoting the number of components for final NMF. 
In this case, ``gsvdnmf`` defaults to augment components on initial NMF solution by 1.

Keyword arguments:

``tol_final``： The tolerence of final NMF, default:``10^{-4}``

``tol_intermediate``: The tolerence of initial NMF (under-complete NMF), default: $\mathrm{tol\\_final}$

Other keyword arguments are passed to ``NMF.nnmf``.

-----

W, H = **gsvdnmf**(X::AbstractMatrix, W::AbstractMatrix, H::AbstractMatrix, [f]::Tuple; 
                   n2 = size(first(f), 2), 
                   tol_final=1e-4, 
                   tol_intermediate=1e-4, 
                   kwargs...)

This function performs "GSVD-NMF" on 2D data matrix ``X`` with customized initialization for initial NMF.

Arguments:

``W``: initialization of initial NMF

``H``: initialization of initial NMF

``f``: SVD (or Truncated SVD) of ``X``

Other arguments and keyword arguments are the same the above method.

-----

Wadd, Hadd, S = **gsvdrecover**(X, W0, H0, kadd; initdata = nothing)

Given NMF solution ``W0`` and ``H``, this function recovers components for NMF solution. 

Outputs:

``Wadd``: augmented NMF solution

``Hadd``: augmented NMF solution

``S``: related generalized singular value

Arguments:

``X``: non-nagetive 2D data matrix

``W0``: NMF solution

``H0``: NMF solution

``kadd``: number of new components

Keyword arguments:

``initdata``: the svd of ``X``

-----

## Citation
The code is welcomed to be used in your publication, please cite:





