# GsvdInitialization

[![CI](https://github.com/HolyLab/GsvdInitialization.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/HolyLab/GsvdInitialization.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/HolyLab/GsvdInitialization.jl/graph/badge.svg?token=LxqRCsZIvn)](https://codecov.io/gh/HolyLab/GsvdInitialization.jl)
[![Aqua QA](https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![version](https://juliahub.com/docs/General/GsvdInitialization/stable/version.svg)](https://juliahub.com/ui/Packages/General/GsvdInitialization)

This package implements GSVD-NMF ([Guo & Holy, *iScience* 2026](https://doi.org/10.1016/j.isci.2026.114708)), a method for recovering missing components in non-negative matrix factorization (NMF).
Starting from a lower-rank factorization `X ≈ W*H`, it proposes new components from the generalized singular value decomposition (GSVD) between the existing factorization and the SVD of `X`, then polishes the augmented factorization with further NMF iterations.
Because components can be added incrementally, GSVD-NMF is convenient and effective for interactive analysis of large-scale data.

See also [NMFMerge](https://github.com/HolyLab/NMFMerge.jl) for the converse operation (merging redundant components). Together, the two substantially improve the quality and consistency of NMF factorizations.

## Installation

GsvdInitialization is a registered package; type `]` at the `julia>` prompt to enter `pkg>` mode and install it with

```
pkg> add GsvdInitialization
```

## Demo

The demo below also uses [NMF.jl](https://github.com/JuliaStats/NMF.jl) and the LinearAlgebra standard library:

```julia
julia> using GsvdInitialization, NMF, LinearAlgebra
```

Generate a ground truth with 10 features (the script ships with the package):

```julia
julia> include(joinpath(pkgdir(GsvdInitialization), "demo", "generate_ground_truth.jl"));

julia> W_GT, H_GT = generate_ground_truth();

julia> X = W_GT * H_GT;
```

<img src="demo/GroundTruth.png" alt="the 10 ground-truth components" width="400"/>

First, run standard NMF on `X`, initialized with NNDSVD. Two precautions aim for the best possible result from NMF alone:

- `maxiter` is set generously (and we verify convergence below), so the run stops at the convergence tolerance, not at the iteration limit;
- NNDSVD is seeded with the full `svd`, which gives higher-quality results than a randomized SVD.

Despite these precautions, the result leaves much to be desired:

```julia
julia> result_nmf = nnmf(X, 10; init=:nndsvd, alg=:cd, tol=1e-4, maxiter=10^4, initdata=svd(X));

julia> result_nmf.converged   # stopped at the tolerance, not the iteration cap
true

julia> sum(abs2, X - result_nmf.W*result_nmf.H) / sum(abs2, X)
0.09999800028665384
```

<img src="demo/ResultHals.png" alt="components found by standard NMF" width="400"/>

The factorization is imperfect: two components are identical, and two features share a single component.
Now run GSVD-NMF on `X` (also initialized with NNDSVD) and compute the new reconstruction error:

```julia
julia> result_gsvd, Λ = gsvdnmf(X, 9 => 10; alg=:cd, tol_final=1e-4, tol_intermediate=1e-2, maxiter=10^4);

julia> W_gsvd, H_gsvd = result_gsvd.W, result_gsvd.H;

julia> sum(abs2, X - W_gsvd*H_gsvd) / sum(abs2, X)
1.2302340443302435e-10
```

An imperfect 9-component factorization was augmented by `gsvdnmf` to an essentially perfect 10-component one.
`Λ` holds the generalized singular values that ranked the candidate augmentation directions, a useful diagnostic of which directions the algorithm chose.
Here are the new components:

<img src="demo/ResultGsvdNMF.png" alt="components found by GSVD-NMF" width="400"/>

## API overview

Complete signatures and doctested examples are in the REPL help (e.g., type `?gsvdnmf`).

- `gsvdnmf(X, n1 => n2; ...)` runs the full pipeline: an initial NMF with `n1` components, augmentation to `n2` components (`n1 < n2 ≤ 2n1`), and a final NMF polish. `gsvdnmf(X, n)` is shorthand for `gsvdnmf(X, n-1 => n)`.
- `gsvdnmf(X, W, H, f; n2, ...)` augments an existing factorization `X ≈ W*H` to `n2` components, using a precomputed SVD `f` of `X`, then polishes with NMF.
- `gsvdrecover(X, W0, H0, kadd, f)` performs the augmentation step alone, adding `kadd` components without the NMF polish.

Each function optionally takes a leading `strategy` argument controlling how the augmented factors are assembled: `GsvdInitialization.truncating` (the default), `GsvdInitialization.joint_nnls`, or a user-supplied callable `(X, W0, H0, Hadd) -> (W_augmented, H_augmented)`.

## Citation

If you use this package, please cite the paper:

> Youdong Guo and Timothy E. Holy, "Recovering missing features in nonnegative matrix factorization via generalized singular value decomposition," *iScience* 29(3):114708 (2026). https://doi.org/10.1016/j.isci.2026.114708

GitHub's "Cite this repository" link (in the About sidebar) provides this in BibTeX and APA formats.
