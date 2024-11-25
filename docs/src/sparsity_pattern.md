# [Improve sparse derivatives](@id sparsity-pattern)

In this tutorial, we show a feature of ADNLPModels.jl to potentially improve the computation of sparse Jacobian and Hessian.

Our test problem is an academic investment control problem:

```math
\begin{aligned}
\min_{u,x} \quad & \int_0^1 (u(t) - 1) x(t) \\
& \dot{x}(t) = \gamma u(t) x(t).
\end{aligned}
```

Using a simple quadrature formula for the objective functional and a forward finite difference for the differential equation, one can obtain a finite-dimensional continuous optimization problem.
One implementation is available in the package [`OptimizationProblems.jl`](https://github.com/JuliaSmoothOptimizers/OptimizationProblems.jl).

```@example ex1
using ADNLPModels
using SparseArrays

T = Float64
n = 100000
N = div(n, 2)
h = 1 // N
x0 = 1
gamma = 3
function f(y; N = N, h = h)
  @views x, u = y[1:N], y[(N + 1):end]
  return 1 // 2 * h * sum((u[k] - 1) * x[k] + (u[k + 1] - 1) * x[k + 1] for k = 1:(N - 1))
end
function c!(cx, y; N = N, h = h, gamma = gamma)
  @views x, u = y[1:N], y[(N + 1):end]
  for k = 1:(N - 1)
    cx[k] = x[k + 1] - x[k] - 1 // 2 * h * gamma * (u[k] * x[k] + u[k + 1] * x[k + 1])
  end
  return cx
end
lvar = vcat(-T(Inf) * ones(T, N), zeros(T, N))
uvar = vcat(T(Inf) * ones(T, N), ones(T, N))
xi = vcat(ones(T, N), zeros(T, N))
lcon = ucon = vcat(one(T), zeros(T, N - 1))

@elapsed begin
  nlp = ADNLPModel!(f, xi, lvar, uvar, [1], [1], T[1], c!, lcon, ucon; hessian_backend = ADNLPModels.EmptyADbackend)
end

```

`ADNLPModel` will automatically prepare an AD backend for computing sparse Jacobian and Hessian.
We disabled the Hessian computation here to focus the measurement on the Jacobian computation.
The keyword argument `show_time = true` can also be passed to the problem's constructor to get more detailed information about the time used to prepare the AD backend.

```@example ex1
using NLPModels
x = sqrt(2) * ones(n)
jac_nln(nlp, x)
```

However, it can be rather costly to determine for a given function the sparsity pattern of the Jacobian and the Hessian of the Lagrangian.
The good news is that determining this pattern a priori can be relatively straightforward, especially for problems like our optimal control investment problem and other problems with differential equations in the constraints.

The following example instantiates the Jacobian backend while manually providing the sparsity pattern.

```@example ex2
using ADNLPModels
using SparseArrays

T = Float64
n = 100000
N = div(n, 2)
h = 1 // N
x0 = 1
gamma = 3
function f(y; N = N, h = h)
  @views x, u = y[1:N], y[(N + 1):end]
  return 1 // 2 * h * sum((u[k] - 1) * x[k] + (u[k + 1] - 1) * x[k + 1] for k = 1:(N - 1))
end
function c!(cx, y; N = N, h = h, gamma = gamma)
  @views x, u = y[1:N], y[(N + 1):end]
  for k = 1:(N - 1)
    cx[k] = x[k + 1] - x[k] - 1 // 2 * h * gamma * (u[k] * x[k] + u[k + 1] * x[k + 1])
  end
  return cx
end
lvar = vcat(-T(Inf) * ones(T, N), zeros(T, N))
uvar = vcat(T(Inf) * ones(T, N), ones(T, N))
xi = vcat(ones(T, N), zeros(T, N))
lcon = ucon = vcat(one(T), zeros(T, N - 1))

@elapsed begin
  Is = Vector{Int}(undef, 4 * (N - 1))
  Js = Vector{Int}(undef, 4 * (N - 1))
  Vs = ones(Bool, 4 * (N - 1))
  for i = 1:(N - 1)
    Is[((i - 1) * 4 + 1):(i * 4)] = [i; i; i; i]
    Js[((i - 1) * 4 + 1):(i * 4)] = [i; i + 1; N + i; N + i + 1]
  end
  J = sparse(Is, Js, Vs, N - 1, n)

  jac_back = ADNLPModels.SparseADJacobian(n, f, N - 1, c!, J)
  nlp = ADNLPModel!(f, xi, lvar, uvar, [1], [1], T[1], c!, lcon, ucon; hessian_backend = ADNLPModels.EmptyADbackend, jacobian_backend = jac_back)
end
```

We recover the same Jacobian.

```@example ex2
using NLPModels
x = sqrt(2) * ones(n)
jac_nln(nlp, x)
```

The same can be done for the Hessian of the Lagrangian.
