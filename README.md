# ADNLPModelss

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
