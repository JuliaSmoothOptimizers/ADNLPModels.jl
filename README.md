# ADNLPModels

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4605982.svg)](https://doi.org/10.5281/zenodo.4605982)
[![GitHub release](https://img.shields.io/github/release/JuliaSmoothOptimizers/ADNLPModels.jl.svg)](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl/releases/latest)
[![](https://img.shields.io/badge/docs-stable-3f51b5.svg)](https://JuliaSmoothOptimizers.github.io/ADNLPModels.jl/stable)
[![](https://img.shields.io/badge/docs-latest-3f51b5.svg)](https://JuliaSmoothOptimizers.github.io/ADNLPModels.jl/dev)
[![codecov](https://codecov.io/gh/JuliaSmoothOptimizers/ADNLPModels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaSmoothOptimizers/ADNLPModels.jl)

![CI](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl/workflows/CI/badge.svg?branch=main)
[![Cirrus CI - Base Branch Build Status](https://img.shields.io/cirrus/github/JuliaSmoothOptimizers/ADNLPModels.jl?logo=Cirrus%20CI)](https://cirrus-ci.com/github/JuliaSmoothOptimizers/ADNLPModels.jl)

This package provides automatic differentiation (AD)-based model implementations that conform to the [NLPModels](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl) API.
The general form of the optimization problem is
```math
\begin{aligned}
\min \quad & f(x) \\
& c_L \leq c(x) \leq c_U \\
& \ell \leq x \leq u,
\end{aligned}
```

## How to Cite

If you use ADNLPModels.jl in your work, please cite using the format given in [CITATION.bib](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl/blob/main/CITATION.bib).

## Installation

<p>
ADNLPModels is a &nbsp;
    <a href="https://julialang.org">
        <img src="https://raw.githubusercontent.com/JuliaLang/julia-logo-graphics/master/images/julia.ico" width="16em">
        Julia Language
    </a>
    &nbsp; package. To install ADNLPModels,
    please <a href="https://docs.julialang.org/en/v1/manual/getting-started/">open
    Julia's interactive session (known as REPL)</a> and press <kbd>]</kbd> key in the REPL to use the package mode, then type the following command
</p>

```julia
pkg> add ADNLPModels
```

## Examples

For optimization in the general form, this package exports two constructors `ADNLPModel` and `ADNLPModel!`.

```julia
using ADNLPModels

f(x) = 100 * (x[2] - x[1]^2)^2 + (x[1] - 1)^2
T = Float64
x0 = T[-1.2; 1.0]
# Rosenbrock
nlp = ADNLPModel(f, x0) # unconstrained

lvar, uvar = zeros(T, 2), ones(T, 2) # must be of same type than `x0`
nlp = ADNLPModel(f, x0, lvar, uvar) # bound-constrained

c(x) = [x[1] + x[2]]
lcon, ucon = -T[0.5], T[0.5]
nlp = ADNLPModel(f, x0, lvar, uvar, c, lcon, ucon) # constrained

c!(cx, x) = begin
  cx[1] = x[1] + x[2]
  return cx
end
nlp = ADNLPModel!(f, x0, lvar, uvar, c!, lcon, ucon) # in-place constrained
```

It is possible to distinguish between linear and nonlinear constraints, see [![](https://img.shields.io/badge/docs-stable-3f51b5.svg)](https://JuliaSmoothOptimizers.github.io/ADNLPModels.jl/stable).

This package also exports the constructors `ADNLSModel` and `ADNLSModel!` for Nonlinear Least Squares (NLS), i.e. when the objective function is a sum of squared terms.

```julia
using ADNLPModels

F(x) = [10 * (x[2] - x[1]^2); x[1] - 1]
nequ = 2 # length of Fx
T = Float64
x0 = T[-1.2; 1.0]
# Rosenbrock in NLS format
nlp = ADNLSModel(F, x0, nequ)
```

The resulting models, `ADNLPModel` and `ADNLSModel`, are instances of `AbstractNLPModel` and implement the NLPModel API, see [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl).

We refer to the documentation for more details on the resulting models, and you can find tutorials on [jso.dev/tutorials/](https://jso.dev/tutorials/) and select the tag `ADNLPModel.jl`.

## AD backend

The following AD packages are supported:

- `ForwardDiff.jl`;
- `ReverseDiff.jl`;

and as optional dependencies (you must load the package before):

- `Enzyme.jl`;
- `Zygote.jl`.

## Bug reports and discussions

If you think you found a bug, feel free to open an [issue](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl/issues).
Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.

If you want to ask a question not suited for a bug report, feel free to start a discussion [here](https://github.com/JuliaSmoothOptimizers/Organization/discussions). This forum is for general discussion about this repository and the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers), so questions about any of our packages are welcome.
