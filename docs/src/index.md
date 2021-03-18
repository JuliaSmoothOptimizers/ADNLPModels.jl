# ADNLPModelss

This package provides a very simple model implement the [NLPModels](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl) API.
It uses [`ForwardDiff`](https://github.com/JuliaDiff/ForwardDiff.jl) to compute the derivatives, which produces dense matrices, so it isn't very efficient for larger problems.

## Install

Install ADNLPModels.jl with the following command.
```julia
pkg> add ADNLPModels
```

## Usage

This package defines two models, [`ADNLPModel`](@ref) for general nonlinear optimization, and [`ADNLSModel`](@ref) other for nonlinear least-squares problems.

```@docs
ADNLPModel
ADNLSModel
```

Check the [Tutorial](@ref) for more details on the usage.

## License

This content is released under the [MPL2.0](https://www.mozilla.org/en-US/MPL/2.0/) License.

## Contents

```@contents
```
