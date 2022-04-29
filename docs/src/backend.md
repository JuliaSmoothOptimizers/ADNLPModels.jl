# How to switch backend in ADNLPModels

`ADNLPModels` allows the use of different backends to compute the derivatives required within NLPModel API.
It uses `ForwardDiff.jl`, `ReverseDiff.jl`, and `Zygote.jl` via optional depencies.

The backend information is in a structure `ADModelBackend` in the attribute `adbackend` of a `ADNLPModel`, it can also be accessed with `get_adbackend`.

```@docs
ADModelBackend
```

The functions currently used to define the NLPModel API and the possible backends are defined in the following table:

| Functions | FowardDiff backends | ReverseDiff backends | Zygote backends |
| ----------- | ----------- | ----------- | ----------- |
| `gradient` and `gradient!` | `ForwardDiffADGradient` | `ReverseDiffADGradient` | `ZygoteADGradient` |
| `jacobian` | `ForwardDiffADJacobian` | `ReverseDiffADJacobian` | `ZygoteADJacobian` |
| `hessian` | `ForwardDiffADHessian` | `ReverseDiffADHessian` | `ZygoteADHessian` |
| `Jprod` | `ForwardDiffADJprod` | `ReverseDiffADJprod` | `ZygoteADJprod` |
| `Jtprod` | `ForwardDiffADJtprod` | `ReverseDiffADJtprod` | `ZygoteADJtprod` |
| `Hvprod` | `ForwardDiffADHvprod` | `ReverseDiffADHvprod` | -- |
| `directional_second_derivative` | `ForwardDiffADGHjvprod` | -- | -- |

The functions `hess_structure!`, `hess_coord!`, `jac_structure!` and `jac_coord!` defined in `ad.jl` are generic to all the backends for now.

## Examples

### Use another backend

Note that the refactorization is non-breaking in the sense that it doesn't change the naive behavior of ADLNPModel.
```
using ADNLPModels, NLPModels
nlp = ADNLPModel(sum, ones(3))
grad(nlp, nlp.meta.x0)
```
Thanks to the splitted backends, we can easily change the backend without recreating an `ReverseDiffAD{T} <: ADBackend` as it is now.
```
using ReverseDiff, ADNLPModels, NLPModels
nlp = ADNLPModel(sum, ones(3), gradient_backend = ADNLPModels.ReverseDiffADGradient)
grad(nlp, nlp.meta.x0)
```
We can then rather easily try out some new implementation for each function.
```
using ADNLPModels, NLPModels
struct NewADGradient <: ADNLPModels.ADBackend end
function NewADGradient(
  nvar::Integer,
  f,
  ncon::Integer = 0;
  kwargs...,
)
  return NewADGradient()
end
ADNLPModels.gradient(adbackend::NewADGradient, f, x) = rand(Float64, size(x))
function ADNLPModels.gradient!(adbackend::NewADGradient, g, f, x)
  g .= rand(Float64, size(x))
  return g
end
nlp = ADNLPModel(sum, ones(3), gradient_backend = NewADGradient)
grad(nlp, nlp.meta.x0)
```

### Change backend

```
adback = ADNLPModels.ADModelBackend(nlp.meta.nvar, nlp.f, gradient_backend = ADNLPModels.ForwardDiffADGradient)
set_adbackend!(nlp, ReverseDiffAD(nlp.meta.nvar, nlp.f))
set_adbackend!(nlp, ReverseDiffAD(nlp.meta.nvar, x -> sum(nlp.F(x) .^ 2)))
set_adbackend!(
  nlp,
  gradient_backend = ADNLPModels.ForwardDiffADGradient,
  jtprod_backend = ADNLPModels.ForwardDiffADJtprod(),
)
```

### Support multiple precision without having to recreate the model

```
using ADNLPModels, NLPModels
nlp = ADNLPModel(sum, ones(3))
grad(nlp, nlp.meta.x0)
set_adbackend!(nlp, gradient_backend = ADNLPModels.ForwardDiffADGradient, x0 = ones(Float32, 3))
grad(nlp, rand(Float32, 3))
```

## Benchmark

** This is the great place for putting benchmark ADNLPModels/CUTEst/JuMP **
