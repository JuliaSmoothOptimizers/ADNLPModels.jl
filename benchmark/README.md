# Benchmarks for ADNLPModels

The problem sets are defined in `problems_sets.jl` and mainly are used scalable problems from [OptimizationProblems.jl](https://github.com/JuliaSmoothOptimizers/OptimizationProblems.jl) with approx. 1000 variables.

## Pkg benchmark

There exist several benchmarks used as package benchmarks, via [`PkgBenchmark.jl`](https://github.com/JuliaCI/PkgBenchmark.jl) and [`BenchmarkCI.jl`](https://github.com/tkf/BenchmarkCI.jl):
- `benchmarks_grad.jl` with the label `run gradient benchmark`: `grad!` from the NLPModel API;
- `benchmarks_Hessian.jl` with the label `run Hessian benchmark`: the initialization of the Hessian backend (which includes the coloring), `hess_coord!` for the objective and Lagrangian, `hess_coord_residual` for NLS problems;
- `benchmarks_Jacobian.jl` with the label `run Jacobian benchmark`: the initialization of the Jacobian backend (which includes the coloring), `jac_coord!`, `jac_coord_residual` for NLS problems;
- `benchmarks_Hessianvector.jl` with the label `run Hessian product benchmark`: `hprod!` for objective and Lagrangian;
- `benchmarks_Jacobianvector.jl` with the label `run Jacobian product benchmark`: `jprod!` and `jtprod!`, as well as `jprod_residual!` and `jtprod_residual!`.

The benchmarks are run whenever the corresponding label is put to the pull request.

## Run backend benchmark and analyze

It is possible to run the benchmark locally with the script `run_local.jl` that will save the results as `jld2` and `json` files.
Then, run `run_analyzer.jl` to get figures comparing the different backends for each sub-benchmark.

## Other ADNLPModels benchmarks

There exist online other benchmarks that concern ADNLPModels:
- [AC Optimal Power Flow](https://discourse.julialang.org/t/ac-optimal-power-flow-in-various-nonlinear-optimization-frameworks/78486): solve an optimization problem with Ipopt and compare various modeling tools;
- [gdalle/SparsityDetectionComparison](https://github.com/gdalle/SparsityDetectionComparison) compares sparsity patterns that are used in Jacobian and Hessian sparsity pattern detection.

If you know other benchmarks, create an issue or open a Pull Request.

## TODOs

- [ ] Add BenchmarkCI push results
- [ ] Automatize and parallelize backend benchmark
- [ ] try/catch to avoid exiting the benchmark on the first error
- [ ] Save the results for each release of ADNLPModels
