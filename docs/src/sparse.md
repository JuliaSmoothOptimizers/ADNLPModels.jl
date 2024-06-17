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

The available backends for sparse derivatives (`SparseADJacobian`, `SparseADHessian` and `SparseReverseADHessian`) have keyword arguments `detector` and `coloring` to specify the sparsity pattern detector and the coloring algorithm, respectively.

- **Detector**: A `detector` must be of type `ADTypes.AbstractSparsityDetector`.
The default detector is `TracerSparsityDetector()` from the package `SparseConnectivityTracer.jl`.
Prior to version 0.8.0, the default detector was `SymbolicSparsityDetector()` from `Symbolics.jl`.

- **Coloring**: A `coloring` must be of type `ADTypes.AbstractColoringAlgorithm`.
The default algorithm is `GreedyColoringAlgorithm()` from the package `SparseMatrixColorings.jl`.

The package [`SparseConnectivityTracer.jl`](https://github.com/adrhill/SparseConnectivityTracer.jl) is used to compute the sparsity pattern of Jacobians and Hessians.
The evaluation of the number of directional derivatives and the seeds required to compute compressed Jacobians and Hessians is performed using [`SparseMatrixColorings.jl`](https://github.com/gdalle/SparseMatrixColorings.jl).
As of release v0.8.1, it has replaced [`ColPack.jl`](https://github.com/exanauts/ColPack.jl).
We acknowledge Guillaume Dalle (@gdalle), Adrian Hill (@adrhill), and Michel Schanen (@michel2323) for the development of these packages.
