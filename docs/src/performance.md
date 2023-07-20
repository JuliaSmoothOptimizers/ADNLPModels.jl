# Performance tips

The package `ADNLPModels.jl` is designed to easily model optimization problems andto allow an efficient access to the [`NLPModel API`](https://github.com/JuliaSmoothOptimizers/NLPModels.jl).
In this tutorial, we will see some tips to ensure the maximum performance of the model.

## Use in-place constructor

When dealing with a constrained optimization problem, it is recommended to use in-place constraint functions.

```@example ex1
using ADNLPModels, NLPModels
f(x) = sum(x)
x0 = ones(2)
lcon = ucon = ones(1)
c_out(x) = [x[1]]
nlp_out = ADNLPModel(f, x0, c_out, lcon, ucon)

c_in(cx, x) = begin
  cx[1] = x[1]
  return cx
end
nlp_in = ADNLPModel!(f, x0, c_in, lcon, ucon)
```

```@example ex1
using BenchmarkTools
cx = rand(1)
x = 18 * ones(2)
@btime cons!(nlp_out, x, cx)
```

```@example ex1
@btime cons!(nlp_in, x, cx)
```

The difference between the two increases with the dimension.

Note that the same applies to nonlinear least squares problems.

```@example ex1
F(x) = [
    x[1];
    x[1] + x[2]^2;
    sin(x[2]);
    exp(x[1] + 0.5)
]
x0 = ones(2)
nequ = 4
nls_out = ADNLSModel(F, x0, nequ)

F!(Fx, x) = begin
  Fx[1] = x[1]
  Fx[2] = x[1] + x[2]^2
  Fx[3] = sin(x[2])
  Fx[4] = exp(x[1] + 0.5)
  return Fx
end
nls_in = ADNLSModel!(F!, x0, nequ)
```

```@example ex1
Fx = rand(4)
@btime residual!(nls_out, x, Fx)
```

```@example ex1
@btime residual!(nls_in, x, Fx)
```

This phenomenon also extends to related backends.

```@example ex1
Fx = rand(4)
v = ones(2)
@btime jprod_residual!(nls_out, x, v, Fx)
```

```@example ex1
@btime jprod_residual!(nls_in, x, v, Fx)
```

## Use only the needed operations

It is tempting to define the most generic and efficient `ADNLPModel` from the start.

```@example ex2
using ADNLPModels, NLPModels, Symbolics
f(x) = (x[1] - x[2])^2
x0 = ones(2)
lcon = ucon = ones(1)
c_in(cx, x) = begin
  cx[1] = x[1]
  return cx
end
nlp = ADNLPModel!(f, x0, c_in, lcon, ucon, show_time = true)
```

However, depending on the size of the problem this might time consuming as initializing each backend takes time.
Besides, some solvers may not require all the API to solve the problem.
For instance, [`Percival.jl`](https://github.com/JuliaSmoothOptimizers/Percival.jl) is matrix-free solver in the sense that it only uses `jprod`, `jtprod` and `hprod`.

```@example ex2
using Percival
stats = percival(nlp)
```

```@example ex2
nlp.counters
```

Therefore, it is more efficient to avoid preparing Jacobian and Hessian backends in this case.

```@example ex2
nlp = ADNLPModel!(f, x0, c_in, lcon, ucon, jacobian_backend = ADNLPModels.EmptyADbackend, hessian_backend = ADNLPModels.EmptyADbackend, show_time = true)
```

or, equivalently, using the `matrix_free` keyword argument

```@example ex2
nlp = ADNLPModel!(f, x0, c_in, lcon, ucon, show_time = true, matrix_free = true)
```

## Benchmarks

This package implements several backends for each method and it is possible to design your own backend as well. 
Then, one way to choose the most efficient one is to run benchmarks.

```@example ex3
using ADNLPModels, NLPModels, OptimizationProblems
```

The package [`OptimizationProblems.jl`](https://github.com/JuliaSmoothOptimizers/OptimizationProblems.jl) provides a collection of optimization problems in JuMP and ADNLPModels syntax.

```@example ex3
meta = OptimizationProblems.meta
scalable_problems = meta[meta.variable_nvar .== true, :name]
```

We select the problems that are scalable, so that there size can be modified. By default, the size is close to `100`.

```@example ex3
using NLPModelsJuMP, Zygote
list_backends = Dict(
  :forward => ADNLPModels.ForwardDiffADGradient,
  :reverse => ADNLPModels.ReverseDiffADGradient,
  :zygote => ADNLPModels.ZygoteADGradient,
)
```

```@example ex3
using DataFrames
nprob = length(scalable_problems)
stats = Dict{Symbol, DataFrame}()
for back in union(keys(list_backends), [:jump])
  stats[back] = DataFrame("name" => scalable_problems,
                 "time" => zeros(nprob),
                 "allocs" => zeros(Int, nprob))
end
```

```@example ex3
using BenchmarkTools
nscal = 1000
for name in scalable_problems
  n = eval(Meta.parse("OptimizationProblems.get_" * name * "_nvar(n = $(nscal))"))
  m = eval(Meta.parse("OptimizationProblems.get_" * name * "_ncon(n = $(nscal))"))
  @info " $(name) with $n vars and $m cons"
  global x = ones(n)
  global g = zeros(n)
  global pb = Meta.parse(name)
  global nlp = MathOptNLPModel(OptimizationProblems.PureJuMP.eval(pb)(n = nscal))
  b = @benchmark grad!(nlp, x, g)
  stats[:jump][stats[:jump].name .== name, :time] = [median(b.times)]
  stats[:jump][stats[:jump].name .== name, :allocs] = [median(b.allocs)]
  for back in keys(list_backends)
    nlp = OptimizationProblems.ADNLPProblems.eval(pb)(n = nscal, gradient_backend = list_backends[back], matrix_free = true)
    b = @benchmark grad!(nlp, x, g)
    stats[back][stats[back].name .== name, :time] = [median(b.times)]
    stats[back][stats[back].name .== name, :allocs] = [median(b.allocs)]
  end
end
```

```@example ex3
using Plots, SolverBenchmark
costnames = ["median time (in ns)", "median allocs"]
costs = [
  df -> df.time,
  df -> df.allocs,
]

gr()

profile_solvers(stats, costs, costnames)
```
