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

Each supported AD package also has its own symbol, such as `:Enzyme` or `:Zygote`, to easily switch between backends.

It is possible to use these pre-defined backends by using the keyword argument `backend` when instantiating the model.

```@example ex1
nlp = ADNLPModel!(f, x0, lvar, uvar, c!, lcon, ucon, backend = :optimized)
get_adbackend(nlp)
```
