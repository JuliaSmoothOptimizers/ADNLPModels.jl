# ADNLPModels

This package provides AD-based model implementations that conform to the [NLPModels](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl) API. The following packages are supported: `ForwardDiff.jl`, `ReverseDiff.jl`, and `Zygote.jl`.

## Install

Install ADNLPModels.jl with the following command.
```julia
pkg> add ADNLPModels
```

## Complementary packages

ADNLPModels.jl functionalities are extended by other packages that are not automatically loaded.
In other words, you sometimes need to load the desired package separately to access some functionalities.

```julia
using ADNLPModels # load only the default functionalities
using Zygote # load the Zygote backends
```

Versions compatibility for the extensions are available in the file `test/Project.toml`.

```@example
print(open(io->read(io, String), "../../test/Project.toml"))
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

## Bug reports and discussions

If you think you found a bug, feel free to open an [issue](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl/issues).
Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.

If you want to ask a question not suited for a bug report, feel free to start a discussion [here](https://github.com/JuliaSmoothOptimizers/Organization/discussions). This forum is for general discussion about this repository and the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers), so questions about any of our packages are welcome.

## Contents

```@contents
```
