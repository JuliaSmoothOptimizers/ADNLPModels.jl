# ADNLPModels

This package provides AD-based model implementations that conform to the [NLPModels](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl) API. The following packages are supported:
- `ForwardDiff.jl`: default choice.
- `Zygote.jl`: you must load `Zygote.jl` separately and pass `ADNLPModels.ZygoteAD` as the `adbackend` keyword argument to the `ADNLPModel` or `ADNLSModel` constructor.
- `ReverseDiff.jl`: you must load `ReverseDiff.jl` separately and pass `ADNLPModels.ReverseDiffAD` as the `adbackend` keyword argument to the `ADNLPModel` or `ADNLSModel` constructor.

## Install

Install ADNLPModels.jl with the following command.
```julia
pkg> add ADNLPModels
```

## Usage

This package defines two models, [`ADNLPModel`](@ref) for general nonlinear optimization, and [`ADNLSModel`](@ref) for nonlinear least-squares problems.

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
