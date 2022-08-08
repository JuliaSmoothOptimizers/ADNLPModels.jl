# ADNLPModels

This package provides AD-based model implementations that conform to the [NLPModels](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl) API. The following packages are supported: `ForwardDiff.jl`, `ReverseDiff.jl`, and `Zygote.jl`.

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
