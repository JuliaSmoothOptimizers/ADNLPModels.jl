# Benchmarks for ADNLPModels

The file utils.jl define some test sets and dispatch the tested backends.
The file additional_backends.jl define some additional backends that are not directly available in ADNLPModels.jl.

## Benchmark NLPModel API

ADNLPModels.jl implements a backend system that several implementation of the NLPModel API.
In all the benchmark using Float64-type the name `jump` refers to models accessed from  JuMP.jl via NLPModelsJuMP.jl.

In this folder, scripts can benchmark:
- `grad!` in benchmark_gradient_ad.jl:
  * "forward": use ForwardDiff.jl with pre-computed Tape;
  * "reverse": use ReverseDiff.jl with pre-computed Tape;
  * "enzyme": use Enzyme.jl.
- `hess_coord!` in benchmark_hessian_ad.jl:
  * "sparse-SDTcol": use SparseDiffTools.jl for coloring and ForwardDiff.jl (sparse);
  * "sparse": use ColPack.jl and ForwardDiff.jl (sparse);
  * "forward": use ForwardDiff.jl (dense).
- `hprod!` in benchmark_hprod_ad.jl and in benchmark_hprod_lag_ad.jl:
  * "forward": use ForwardDiff.jl;
  * "sdtforward": use SparseDiffTools.jl in forward mode
  * "reverse": use ReverseDiff.jl.
- `jac_coord!` in benchmark_jacobian_ad.jl:
  * "sparse-SDTcol": use SparseDiffTools.jl for coloring and ForwardDiff.jl (sparse);
  * "sparse": use ColPack.jl and ForwardDiff.jl (sparse);
  * "SDTsparse": use SparseDiffTools.jl (sparse);
  * "forward": use ForwardDiff.jl (dense);
  * "reverse": use ReverseDiff.jl (dense);
  * "zygote": use Zygote.jl (dense).
- `jprod!` in benchmark_jprod_ad.jl:
  * "forward": use ForwardDiff.jl;
  * "reverse": use ReverseDiff.jl;
  * "sdtforward": use SparseDiffTools.jl in forward mode.
- `jtprod!` in benchmark_jtprod_ad.jl:
  * "forward": use ForwardDiff.jl;
  * "reverse": use ReverseDiff.jl.

## Benchmark coloring

coloration.jl benchmarks several algorithms from the packages ColPack.jl and SparseDiffTools.jl to compute coloring.

## Naming convention for the results

```julia
{date}_adnlpmodels_benchmark_{function benchmarked}_{optimized or generic}_{size info}_{typing}.jld2
```

Some comments:
- A `mono` typing indicates that only `Float64` is used;
- `nscal_100` indicates that scalable problems are used with `n=100`;
- `generic` stands for backend that allow several types, while `optimized` is typed with the `x0` given to the `ADNLPModel` constructor.

# TODOs

- [ ] Parallelize benchmark
- [ ] try/catch to avoid exiting the benchmark on first error
- [ ] add scripts to run benchmarks