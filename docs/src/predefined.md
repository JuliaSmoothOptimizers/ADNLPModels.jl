# Default backend and performance in ADNLPModels

As illustrated in the tutorial on backends, `ADNLPModels.jl` use different backend for each method from the `NLPModel API` that are implemented.
By default, it uses the following:
```@example ex1
using ADNLPModels, NLPModels

f(x) = 100 * (x[2] - x[1]^2)^2 + (x[1] - 1)^2
T = Float64
x0 = T[-1.2; 1.0]
lvar, uvar = zeros(T, 2), ones(T, 2) # must be of same type than `x0`
lcon, ucon = -T[0.5], T[0.5]
c!(cx, x) = begin
  cx[1] = x[1] + x[2]
  return cx
end
nlp = ADNLPModel!(f, x0, lvar, uvar, c!, lcon, ucon)
get_adbackend(nlp)
```

Note that `ForwardDiff.jl` is mainly used as it is efficient and stable.

## Predefined backends

Another way to know the default backends used is to check the constant `ADNLPModels.default_backend`.
```@example ex1
ADNLPModels.default_backend
```

More generally, the package anticipates more uses
```@example ex1
ADNLPModels.predefined_backend
```

The backend `:optimized` will mainly focus on the most efficient approaches, for instance using `ReverseDiff` to compute the gradient instead of `ForwardDiff`.

```@example ex1
ADNLPModels.predefined_backend[:optimized]
```

The backend `:generic` focuses on backend that make no assumptions on the element type, see [Creating an ADNLPModels backend that supports multiple precisions](https://jso.dev/tutorials/generic-adnlpmodels/).

It is possible to use these pre-defined backends using the keyword argument `backend` when instantiating the model.

```@example ex1
nlp = ADNLPModel!(f, x0, lvar, uvar, c!, lcon, ucon, backend = :optimized)
get_adbackend(nlp)
```

## Hessian and Jacobian computations

It is to be noted that by default the Jacobian and Hessian matrices are sparse.

```@example ex1
(get_nnzj(nlp), get_nnzh(nlp))  # number of nonzeros elements in the Jacobian and Hessian
```

```@example ex1
f(x) = (x[1] - 1)^2
T = Float64
x0 = T[-1.2; 1.0]
lvar, uvar = zeros(T, 2), ones(T, 2) # must be of same type than `x0`
lcon, ucon = -T[0.5], T[0.5]
c!(cx, x) = begin
  cx[1] = x[2]
  return cx
end
nlp = ADNLPModel!(f, x0, lvar, uvar, c!, lcon, ucon, backend = :optimized)
(get_nnzj(nlp), get_nnzh(nlp))
```

```@example ex1
x = rand(T, 2)
jac(nlp, x)
```

The package [`SparseConnectivityTracer.jl`](https://github.com/adrhill/SparseConnectivityTracer.jl) is used to compute the sparsity pattern of Jacobians and Hessians.
The evaluation of the number of directional derivatives and the seeds needed to evaluate the compressed Jacobians and Hessians is done by [`ColPack.jl`](https://github.com/exanauts/ColPack.jl).
We acknowledge Guillaume Dalle (@gdalle), Adrian Hill (@adrhill), and Michel Schanen (@michel2323) for the development of these packages.
