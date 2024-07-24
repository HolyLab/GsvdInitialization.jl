# GsvdInitialization

[![CI](https://github.com/HolyLab/GsvdInitialization.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/HolyLab/GsvdInitialization.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/HolyLab/GsvdInitialization.jl/graph/badge.svg?token=LxqRCsZIvn)](https://codecov.io/gh/HolyLab/GsvdInitialization.jl)

This package includes the code of the paper 'GSVD-NMF: Recovering Missing Features in
Non-negative Matrix Factorization`. 
It is used to recover Non-negative matrix factorization(NMF) components from low-dimensional space to higher dimensional space by exploiting the generalized singular value decomposition (GSVD) between existing NMF results and the SVD of X.
This method allows the incremental expansion of the number of components, which can be convenient and effective for interactive analysis of large-scale data.

---------------------------

Demo:

Install the package
```julia
julia>] add GsvdInitialization;
```




---------------------------

## Functions

**gsvdnmf**(X, ncomponents::Pair{Int,Int}; tol_final=1e-4, tol_intermediate=tol_final, W0=nothing, H0=nothing, kwargs...)
This function performs "GSVD-NMF" on 2D data matrix ``X``.

Arguments:

``ncomponents::Pair{Int,Int}``: in the form of ``n1 => n2``, augment from ``n1`` components to ``n2``components, where ``n1`` is the number of components for initial NMF (under-complete NMF), and ``n2`` is the number of components for final NMF.

Alternatively, ``ncomponents`` can be an integer denoting the initial number of components (under-complete NMF). In this case, ``gsvdnmf`` defaults to augment components on initial NMF solution by 1.

Keyword arguments:

``tol_final``： The tolerence of final NMF, default:``10^{-4}``

``tol_intermediate``: The tolerence of initial NMF (under-complete NMF), default: $\mathrm{tol\\_final}$

``W0``: initialization of initial NMF, default: ``nothing``

``H0``: initialization of initial NMF, default: ``nothing``

If one of ``W0`` and ``H0`` is ``nothing``, NNDSVD is used for initialization.

Other keywords arguments are passed to ``NMF.nnmf``.

-----

**gsvdrecover**(X, W0, H0, kadd; initdata = nothing)

Arguments:

``X``: 

``W0``: 

``H0``: 

``kadd``: 

Keyword arguments:

``initdata``: 

-----

## Citation
The code is welcomed to be used in your publication, please cite:





