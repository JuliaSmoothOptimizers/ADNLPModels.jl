# ADNLPModels

This package provides a very simple model implement the [NLPModels](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl) API.
It uses [`ForwardDiff`](https://github.com/JuliaDiff/ForwardDiff.jl) to compute the derivatives, which produces dense matrices, so it isn't very efficient for larger problems.

## How to Cite

If you use ADNLPModels.jl in your work, please cite using the format given in [CITATION.bib](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl/blob/master/CITATION.bib).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4605982.svg)](https://doi.org/10.5281/zenodo.4605982)
[![GitHub release](https://img.shields.io/github/release/JuliaSmoothOptimizers/ADNLPModels.jl.svg)](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl/releases/latest)
[![](https://img.shields.io/badge/docs-stable-3f51b5.svg)](https://JuliaSmoothOptimizers.github.io/ADNLPModels.jl/stable)
[![](https://img.shields.io/badge/docs-latest-3f51b5.svg)](https://JuliaSmoothOptimizers.github.io/ADNLPModels.jl/dev)
[![codecov](https://codecov.io/gh/JuliaSmoothOptimizers/ADNLPModels.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaSmoothOptimizers/ADNLPModels.jl)

![CI](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl/workflows/CI/badge.svg?branch=master)
[![Cirrus CI - Base Branch Build Status](https://img.shields.io/cirrus/github/JuliaSmoothOptimizers/ADNLPModels.jl?logo=Cirrus%20CI)](https://cirrus-ci.com/github/JuliaSmoothOptimizers/ADNLPModels.jl)

## Installation

```julia
pkg> add ADNLPModels
```

## TODO - sparse part
- generic problems for ADNLPModel, RADNLPModel, etc...

### Tests
- get tests from NLPModels ✓
- new test problems with funny structure ✓
- pick matrix in a depot and generate quadratic problems
### Benchmark
- improve the output of benchmark function ✓
- improve the benchmark or creation of models so that we can compare intra-RADNLP ✓
- compare reversediff to compute the grad! -> add pre-allocations ✓
- and zygote (bug?)
### Code (1st goal is for unconstrained)
- improve constructors ✓
- compute nnzh, hess_structure! and hess_coord!
- grad! ✓
- hprod!
- constructors for bound-constrained
- Uncomment consistency.jl and runtests.jl lines as we get a first implementation.
### Code (2nt goal is constrained)
- compute nnzh and nnzj
- cons!, jac_structure!, jac_coord!, jprod, jtprod, jac_op
- hess_coord!, hprod!

## Debate
- Have different models or one with options ?
  The advantage of having options is the possibility to easily change the behavior
  of an NLPModels during the execution of an algorithms.

  However, it should be slower ?
- Is it an AbstractNLPModel or an AbstractADNLPModel?