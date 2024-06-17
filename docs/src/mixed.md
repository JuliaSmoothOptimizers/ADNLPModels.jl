# Build an hybrid NLPModel

The package `ADNLPModels.jl` implements the [`NLPModel API`](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) using automatic differentiation (AD) backends.
It is also possible to build hybrid models that use AD to complete the implementation of a given `NLPModel`.

In the following example, we use [`ManualNLPModels.jl`](https://github.com/JuliaSmoothOptimizers/ManualNLPModels.jl) to build an NLPModel with the gradient and the Jacobian functions implemented.

```@example ex1
using ManualNLPModels
f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
g!(gx, x) = begin
  y1, y2 = x[1] - 1, x[2] - x[1]^2
  gx[1] = 2 * y1 - 16 * x[1] * y2
  gx[2] = 8 * y2
  return gx
end

c!(cx, x) = begin
  cx[1] = x[1] + x[2]
  return cx
end
j!(vals, x) = begin
  vals[1] = 1.0
  vals[2] = 1.0
  return vals
end

x0 = [-1.2; 1.0]
model = NLPModel(
  x0,
  f,
  grad = g!,
  cons = (c!, [0.0], [0.0]),
  jac_coord = ([1; 1], [1; 2], j!),
)
```

However, methods involving the Hessian or Jacobian-vector products are not implemented.

```@example ex1
using NLPModels
v = ones(2)
try
  jprod(model, x0, v)
catch e
  println("$e")
end
```

This is where building hybrid models with `ADNLPModels.jl` becomes useful.

```@example ex1
using ADNLPModels
nlp = ADNLPModel!(model, gradient_backend = model, jacobian_backend = model)
```

This would be equivalent to do.
```julia
nlp = ADNLPModel!(
  f,
  x0,
  c!,
  [0.0],
  [0.0],
  gradient_backend = model,
  jacobian_backend = model,
)
```

```@example ex1
get_adbackend(nlp)
```

Note that the backends used for the gradient and jacobian are now `NLPModel`. So, a call to `grad` on `nlp`

```@example ex1
grad(nlp, x0)
```

would call `grad` on `model`

```@example ex1
neval_grad(model)
```

Moreover, as expected, the ADNLPModel `nlp` also implements the missing methods, e.g.

```@example ex1
jprod(nlp, x0, v)
```
