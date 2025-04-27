# [Sparse Hessian and Jacobian computations](@id sparse)

By default, the Jacobian and Hessian are treated as sparse.

```@example ex1
using ADNLPModels, NLPModels

f(x) = (x[1] - 1)^2
T = Float64
x0 = T[-1.2; 1.0]
nvar, ncon = 2, 1
lvar, uvar = zeros(T, nvar), ones(T, nvar)
lcon, ucon = -T[0.5], T[0.5]
c!(cx, x) = begin
  cx[1] = x[2]
  return cx
end
nlp = ADNLPModel!(f, x0, lvar, uvar, c!, lcon, ucon, backend = :optimized)
```

```@example ex1
(get_nnzj(nlp), get_nnzh(nlp))  # Number of nonzero elements in the Jacobian and Hessian
```

```@example ex1
x = rand(T, nvar)
J = jac(nlp, x)
```

```@example ex1
x = rand(T, nvar)
H = hess(nlp, x)
```

## Options for sparsity pattern detection and coloring

The backends available for sparse derivatives (`SparseADJacobian`, `SparseEnzymeADJacobian`, `SparseADHessian`, `SparseReverseADHessian`, and `SparseEnzymeADHessian`) allow for customization through keyword arguments such as `detector` and `coloring_algorithm`.
These arguments specify the sparsity pattern detector and the coloring algorithm, respectively.

- A **`detector`** must be of type `ADTypes.AbstractSparsityDetector`.
  The default detector is `TracerSparsityDetector()` from the package `SparseConnectivityTracer.jl`.
  Prior to version 0.8.0, the default was `SymbolicSparsityDetector()` from `Symbolics.jl`.
  A `TracerLocalSparsityDetector()` is also available and can be used if the sparsity pattern of Jacobians and Hessians depends on `x`.

```@example ex1
import SparseConnectivityTracer.TracerLocalSparsityDetector

set_adbackend!(
  nlp,
  jacobian_backend = ADNLPModels.SparseADJacobian(nvar, f, ncon, c!, detector=TracerLocalSparsityDetector()),
  hessian_backend = ADNLPModels.SparseADHessian(nvar, f, ncon, c!, detector=TracerLocalSparsityDetector()),
)
```

- A **`coloring_algorithm`** must be of type `SparseMatrixColorings.GreedyColoringAlgorithm`.
  The default algorithm is `GreedyColoringAlgorithm{:direct}()` for `SparseADJacobian`, `SparseEnzymeADJacobian` and `SparseADHessian`, while it is `GreedyColoringAlgorithm{:substitution}()` for `SparseReverseADHessian` and `SparseEnzymeADHessian`.
  These algorithms are provided by the package `SparseMatrixColorings.jl`.

```@example ex1
using SparseMatrixColorings

set_adbackend!(
  nlp,
  hessian_backend = ADNLPModels.SparseADHessian(nvar, f, ncon, c!, coloring_algorithm=GreedyColoringAlgorithm{:substitution}()),
)
```

The `GreedyColoringAlgorithm{:direct}()` performs column coloring for Jacobians and star coloring for Hessians.
In contrast, `GreedyColoringAlgorithm{:substitution}()` applies acyclic coloring for Hessians. The `:substitution` mode generally requires fewer colors than `:direct`, thus fewer directional derivatives are needed to reconstruct the sparse Hessian.
However, it necessitates storing the compressed sparse Hessian, while `:direct` coloring only requires storage for one column of the compressed Hessian.

The `:direct` coloring mode is numerically more stable and may be preferable for highly ill-conditioned Hessians, as it avoids solving triangular systems to compute nonzero entries from the compressed Hessian.

## Extracting sparsity patterns

`ADNLPModels.jl` provides the function [`get_sparsity_pattern`](@ref) to retrieve the sparsity patterns of the Jacobian or Hessian from a model.

```@example ex3
using SparseArrays, ADNLPModels, NLPModels

nvar = 10
ncon = 5

f(x) = sum((x[i] - i)^2 for i = 1:nvar) + x[nvar] * sum(x[j] for j = 1:nvar-1)

function c!(cx, x)
  cx[1] = x[1] + x[2]
  cx[2] = x[1] + x[2] + x[3]
  cx[3] = x[2] + x[3] + x[4]
  cx[4] = x[3] + x[4] + x[5]
  cx[5] = x[4] + x[5]
  return cx
end

T = Float64
x0 = -ones(T, nvar)
lvar = zeros(T, nvar)
uvar = 2 * ones(T, nvar)
lcon = -0.5 * ones(T, ncon)
ucon = 0.5 * ones(T, ncon)

nlp = ADNLPModel!(f, x0, lvar, uvar, c!, lcon, ucon)
```
```@example ex3
J = get_sparsity_pattern(nlp, :jacobian)
```
```@example ex3
H = get_sparsity_pattern(nlp, :hessian)
```

## Using known sparsity patterns

If the sparsity pattern of the Jacobian or the Hessian is already known, you can provide it directly.
This may happen when the pattern is derived from the application or has been computed previously and saved for reuse.
Note that both the lower and upper triangular parts of the Hessian are required during the coloring phase.

```@example ex2
using SparseArrays, ADNLPModels, NLPModels

nvar = 10
ncon = 5

f(x) = sum((x[i] - i)^2 for i = 1:nvar) + x[nvar] * sum(x[j] for j = 1:nvar-1)

H = SparseMatrixCSC{Bool, Int}(
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

J = SparseMatrixCSC{Bool, Int}(
  [ 1  1  0  0  0  0  0  0  0  0 ;
    1  1  1  0  0  0  0  0  0  0 ;
    0  1  1  1  0  0  0  0  0  0 ;
    0  0  1  1  1  0  0  0  0  0 ;
    0  0  0  1  1  0  0  0  0  0 ]
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

The section ["providing the sparsity pattern for sparse derivatives"](@ref sparsity-pattern) illustrates this feature with a more advanced application.

## Automatic sparse differentiation (ASD)

For a deeper understanding of how `ADNLPModels.jl` computes sparse Jacobians and Hessians, you can refer to the following blog post: ["An Illustrated Guide to Automatic Sparse Differentiation"](https://iclr-blogposts.github.io/2025/blog/sparse-autodiff/).
It explains the key ideas behind sparse automatic differentiation (ASD), and why this approach is critical for large-scale nonlinear optimization.

### Acknowledgements

The package [`SparseConnectivityTracer.jl`](https://github.com/adrhill/SparseConnectivityTracer.jl) is used to compute the sparsity pattern of Jacobians and Hessians.
The evaluation of the number of directional derivatives and the seeds required to compute compressed Jacobians and Hessians is performed using [`SparseMatrixColorings.jl`](https://github.com/gdalle/SparseMatrixColorings.jl).
As of release v0.8.1, it has replaced [`ColPack.jl`](https://github.com/exanauts/ColPack.jl).
We acknowledge Guillaume Dalle (@gdalle), Adrian Hill (@adrhill), Alexis Montoison (@amontoison), and Michel Schanen (@michel2323) for the development of these packages.
