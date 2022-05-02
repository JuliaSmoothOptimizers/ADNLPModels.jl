# How to switch backend in ADNLPModels

`ADNLPModels` allows the use of different backends to compute the derivatives required within NLPModel API.
It uses `ForwardDiff.jl`, `ReverseDiff.jl`, and `Zygote.jl` via optional depencies.

The backend information is in a structure [`ADNLPModels.ADModelBackend`](@ref) in the attribute `adbackend` of a `ADNLPModel`, it can also be accessed with [`get_adbackend`](@ref).

The functions used internally to define the NLPModel API and the possible backends are defined in the following table:

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

We now present a serie of practical examples. For simplicity, we focus here on unconstrained optimization problem. All these examples can be generalized to problems with bounds, constraints or nonlinear least-squares.

### Use another backend

As shown in [Tutorial](@ref), it is very straightforward to instantiate an `ADNLPModel` using an objective function and an initial guess.
```@example adnlp
using ADNLPModels, NLPModels
f(x) = sum(x)
x0 = ones(3)
nlp = ADNLPModel(f, x0)
grad(nlp, nlp.meta.x0) # returns the gradient at x0
```
Thanks to the backends inside `ADNLPModels.jl`, it is easy to change the backend for one (or more) function using the `kwargs` presented in the table above.
```@example adnlp
nlp = ADNLPModel(f, x0, gradient_backend = ADNLPModels.ReverseDiffADGradient)
grad(nlp, nlp.meta.x0) # returns the gradient at x0 using `ReverseDiff`
```
It is also possible to try some new implementation for each function. First, we define a new `ADBackend` structure.
```@example adnlp
struct NewADGradient <: ADNLPModels.ADBackend end
function NewADGradient(
  nvar::Integer,
  f,
  ncon::Integer = 0;
  kwargs...,
)
  return NewADGradient()
end
```
Then, we implement the desired functions following the table above.
```@example adnlp
ADNLPModels.gradient(adbackend::NewADGradient, f, x) = rand(Float64, size(x))
function ADNLPModels.gradient!(adbackend::NewADGradient, g, f, x)
  g .= rand(Float64, size(x))
  return g
end
```
Finally, we use the homemade backend to compute the gradient.
```@example adnlp
nlp = ADNLPModel(sum, ones(3), gradient_backend = NewADGradient)
grad(nlp, nlp.meta.x0) # returns the gradient at x0 using `NewADGradient`
```

### Change backend

Once an instance of an `ADNLPModel` has been created, it is possible to change the backends without re-instantiating the model.
```@example adnlp2
using ADNLPModels, NLPModels
f(x) = 100 * (x[2] - x[1]^2)^2 + (x[1] - 1)^2
x0 = 3 * ones(2)
nlp = ADNLPModel(f, x0)
get_adbackend(nlp) # returns the `ADModelBackend` structure that regroup all the various backends.
```
There are currently two ways to modify instantiated backends. The first one is to instantiate a new `ADModelBackend` and use `set_adbackend!` to modify `nlp`.
```@example adnlp2
adback = ADNLPModels.ADModelBackend(nlp.meta.nvar, nlp.f, gradient_backend = ADNLPModels.ForwardDiffADGradient)
set_adbackend!(nlp, adback)
get_adbackend(nlp)
```
The alternative is to use ``set_adbackend!` and pass the new backends via `kwargs`. In the second approach, it is possible to pass either the type of the desired backend or an instance as shown below.
```@example adnlp2
set_adbackend!(
  nlp,
  gradient_backend = ADNLPModels.ForwardDiffADGradient,
  jtprod_backend = ADNLPModels.ForwardDiffADJtprod(),
)
get_adbackend(nlp)
```

### Support multiple precision without having to recreate the model

One of the strength of `ADNLPModels.jl` is the type flexibility. Let's assume, we first instantiate an `ADNLPModel` with a `Float64` initial guess.
```@example adnlp3
using ADNLPModels, NLPModels
f(x) = 100 * (x[2] - x[1]^2)^2 + (x[1] - 1)^2
x0 = 3 * ones(2) # Float64 initial guess
nlp = ADNLPModel(f, x0)
```
Then, the gradient will return a vector of `Float64`.
```@example adnlp3
x64 = rand(2)
grad(nlp, x64)
```
It is now possible to move to a different type, for instance `Float32`, while keeping the instance `nlp`.
```@example adnlp3
x0_32 = ones(Float32, 2)
set_adbackend!(nlp, gradient_backend = ADNLPModels.ForwardDiffADGradient, x0 = x0_32)
x32 = rand(Float32, 2)
grad(nlp, x32)
```
