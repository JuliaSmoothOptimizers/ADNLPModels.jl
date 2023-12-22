# Improve sparse derivatives

In this tutorial, we show a simple trick to dramatically improve the computation of sparse Jacobian and Hessian matrices.

Our test problem is an academic investment control problem:

```math
\begin{aligned}
\min_{u,x} \quad & \int_0^1 (u(t) - 1) x(t) \\
& \dot{x}(t) = \gamma u(t) x(t).
\end{aligned}
```

Using a simple quadrature formula for the objective functional and a forward finite difference for the differential equation, one can obtain a finite-dimensional continuous optimisation problem.
One is implementation is available in the package [`OptimizationProblems.jl`](https://github.com/JuliaSmoothOptimizers/OptimizationProblems.jl).

```@example ex1
using ADNLPModels
using OptimizationProblems
using Symbolics
using SparseArrays

n = 1000000
@elapsed begin
  nlp = OptimizationProblems.ADNLPProblems.controlinvestment(n = n, hessian_backend = ADNLPModels.EmptyADbackend)
end

```

After adding the package `Symbolics.jl`, the `ADNLPModel` will automatically try to prepare AD-backend to compute sparse Jacobian and Hessian.
We disabled the Hessian computation here to focus the measurement on the Jacobian computation.
The keyword argument `show_time = true` can also be passed to the problem's constructor to get more detailed information about the time used to prepare the AD backend.

```@example ex1
using NLPModels
x = sqrt(2) * ones(n)
jac_nln(nlp, x)
```

However, it can be rather costly to determine for a given function the sparsity pattern of the Jacobian and the Lagrangian Hessian matrices.
The good news is that it can be quite easy to have a good approximation of this sparsity pattern dealing with problems like our optimal control investment problem, and problem with differential equations in the constraints in general.

The following example specialize the function `compute_jacobian_sparsity` to manually provide the sparsity pattern.

```@example ex2
using ADNLPModels
using OptimizationProblems
using Symbolics
using SparseArrays

n = 1000000
N = div(n, 2)

function ADNLPModels.compute_jacobian_sparsity(c!, cx, x0; n = n, N = N)
  # S = Symbolics.jacobian_sparsity(c!, cx, x0)
  S = spzeros(Bool, N - 1, n)
  for i =1:(N - 1)
    S[i, i] = true
    S[i, i + 1] = true
    S[i, N + i] = true
    S[i, N + i + 1] = true
  end
  return S
end

@elapsed begin
  nlp = OptimizationProblems.ADNLPProblems.controlinvestment(n = n, hessian_backend = ADNLPModels.EmptyADbackend)
end
```

A similar Jacobian matrix is obtained at a lower price.

```@example ex2
using NLPModels
x = sqrt(2) * ones(n)
jac_nln(nlp, x)
```

The function `compute_hessian_sparsity(f, nvar, c!, ncon)` does the same for the Lagrangian Hessian.
