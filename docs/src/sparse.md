# Sparse Hessian and Jacobian computations

It is to be noted that by default the Jacobian and Hessian are sparse.

```@example ex1
using ADNLPModels, NLPModels

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
```

```@example ex1
(get_nnzj(nlp), get_nnzh(nlp))  # number of nonzeros elements in the Jacobian and Hessian
```

```@example ex1
x = rand(T, 2)
J = jac(nlp, x)
```

```@example ex1
x = rand(T, 2)
H = hess(nlp, x)
```

The available backends for sparse derivatives (`SparseADJacobian`, `SparseADHessian` and `SparseReverseADHessian`) have keyword arguments `detector` and `coloring_algorithm` to specify the sparsity pattern detector and the coloring algorithm, respectively.

- A **`detector`** must be of type `ADTypes.AbstractSparsityDetector`.
The default detector is `TracerSparsityDetector()` from the package `SparseConnectivityTracer.jl`.
Prior to version 0.8.0, the default detector was `SymbolicSparsityDetector()` from `Symbolics.jl`.

- A **`coloring_algorithm`** must be of type `SparseMatrixColorings.GreedyColoringAlgorithm`.
The default algorithm is `GreedyColoringAlgorithm{:direct}()` for `SparseADJacobian` and `SparseADHessian`, while it is `GreedyColoringAlgorithm{:substitution}()` for `SparseReverseADHessian`.
These algorithms are available in the package `SparseMatrixColorings.jl`.

The `GreedyColoringAlgorithm{:direct}()` performs column coloring for Jacobians and star coloring for Hessians.
In contrast, `GreedyColoringAlgorithm{:substitution}()` applies acyclic coloring for Hessians.
The `:substitution` coloring mode usually finds fewer colors than the `:direct` mode and thus fewer directional derivatives are needed to recover all non-zeros of the sparse Hessian.
However, it requires storing the compressed sparse Hessian, while `:direct` coloring only stores one column of the compressed Hessian.

The `:direct` coloring mode is numerically more stable and may be preferable for highly ill-conditioned Hessian as it doesn't require solving triangular systems to compute the non-zeros from the compressed Hessian.

If the sparsity pattern of the Jacobian of the constraint or the Hessian of the Lagrangian is available, you can directly provide them.
```@example ex2
using SparseArrays, ADNLPModels, NLPModels

nvar = 10
ncon = 5

f(x) = sum((x[i] - i)^2 for i = 1:nvar) + x[nvar] * sum(x[j] for j = 1:nvar-1)

H = SparseMatrixCSC{Bool, Int64}(
[ 1  0  0  0  0  0  0  0  0  1 ;
  0  1  0  0  0  0  0  0  0  1 ;
  0  0  1  0  0  0  0  0  0  1 ;
  0  0  0  1  0  0  0  0  0  1 ;
  0  0  0  0  1  0  0  0  0  1 ;
  0  0  0  0  0  1  0  0  0  1 ;
  0  0  0  0  0  0  1  0  0  1 ;
  0  0  0  0  0  0  0  1  0  1 ;
  0  0  0  0  0  0  0  0  1  1 ;
  1  1  1  1  1  1  1  1  1  1 ]
)

function c!(cx, x)
  cx[1] = x[1] + x[2]
  cx[2] = x[1] + x[2] + x[3]
  cx[3] = x[2] + x[3] + x[4]
  cx[4] = x[3] + x[4] + x[5]
  cx[5] = x[4] + x[5]
  return cx
end

J = SparseMatrixCSC{Bool, Int64}(
[ 1  1  0  0  0 ;
  1  1  1  0  0 ;
  0  1  1  1  0 ;
  0  0  1  1  1 ;
  0  0  0  1  1 ]
)

T = Float64
x0 = -ones(T, nvar)
lvar = zeros(T, nvar)
uvar = 2 * ones(T, nvar)
lcon = -0.5 * ones(T, ncon)
ucon = 0.5 * ones(T, ncon)

J_backend = ADNLPModels.SparseADJacobian(nvar, f, ncon, c!, J)
H_backend = ADNLPModels.SparseADHessian(nvar, f, ncon, c!, H)

nlp = ADNLPModel!(f, x0, lvar, uvar, c!, lcon, ucon, jacobian_backend=J_backend, hessian_backend=H_backend)
```

The package [`SparseConnectivityTracer.jl`](https://github.com/adrhill/SparseConnectivityTracer.jl) is used to compute the sparsity pattern of Jacobians and Hessians.
The evaluation of the number of directional derivatives and the seeds required to compute compressed Jacobians and Hessians is performed using [`SparseMatrixColorings.jl`](https://github.com/gdalle/SparseMatrixColorings.jl).
As of release v0.8.1, it has replaced [`ColPack.jl`](https://github.com/exanauts/ColPack.jl).
We acknowledge Guillaume Dalle (@gdalle), Adrian Hill (@adrhill), Alexis Montoison (@amontoison), and Michel Schanen (@michel2323) for the development of these packages.
