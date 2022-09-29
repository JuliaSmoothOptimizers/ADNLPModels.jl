# ADNLPModels

This package provides AD-based model implementations that conform to the [NLPModels](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl) API. The following packages are supported:
- `ForwardDiff.jl`: default choice.
- `Zygote.jl`: you must load `Zygote.jl` separately and pass `ADNLPModels.ZygoteAD()` as the `adbackend` keyword argument to the `ADNLPModel` or `ADNLSModel` constructor.
- `ReverseDiff.jl`: you must load `ReverseDiff.jl` separately and pass `ADNLPModels.ReverseDiffAD()` as the `adbackend` keyword argument to the `ADNLPModel` or `ADNLSModel` constructor.

## How to Cite

If you use ADNLPModels.jl in your work, please cite using the format given in [CITATION.bib](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl/blob/main/CITATION.bib).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4605982.svg)](https://doi.org/10.5281/zenodo.4605982)
[![GitHub release](https://img.shields.io/github/release/JuliaSmoothOptimizers/ADNLPModels.jl.svg)](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl/releases/latest)
[![](https://img.shields.io/badge/docs-stable-3f51b5.svg)](https://JuliaSmoothOptimizers.github.io/ADNLPModels.jl/stable)
[![](https://img.shields.io/badge/docs-latest-3f51b5.svg)](https://JuliaSmoothOptimizers.github.io/ADNLPModels.jl/dev)
[![codecov](https://codecov.io/gh/JuliaSmoothOptimizers/ADNLPModels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaSmoothOptimizers/ADNLPModels.jl)

![CI](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl/workflows/CI/badge.svg?branch=main)
[![Cirrus CI - Base Branch Build Status](https://img.shields.io/cirrus/github/JuliaSmoothOptimizers/ADNLPModels.jl?logo=Cirrus%20CI)](https://cirrus-ci.com/github/JuliaSmoothOptimizers/ADNLPModels.jl)

## Installation

```julia
pkg> add ADNLPModels
```

# Bug reports and discussions

If you think you found a bug, feel free to open an [issue](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl/issues).
Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.

If you want to ask a question not suited for a bug report, feel free to start a discussion [here](https://github.com/JuliaSmoothOptimizers/Organization/discussions). This forum is for general discussion about this repository and the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers), so questions about any of our packages are welcome.
