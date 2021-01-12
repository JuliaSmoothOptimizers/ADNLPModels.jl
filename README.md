# ADNLPModels

## TODO
- generic problems for ADNLPModel, RADNLPModel, etc...

### Tests
- get tests from NLPModels ✓
- new test problems with funny structure ✓
- pick matrix in a depot and generate quadratic problems
### Benchmark
- improve the output of benchmark function
- compare reversediff and zygote to compute the grad!
### Code (1st goal is for unconstrained)
- compute nnzh, hess_structure! and hess_coord!
- grad!
- hprod!
- constructors for bound-constrained
- Uncomment consistency.jl and runtests.jl lines as we get a first implementation.
### Code (2nt goal is constrained)
- compute nnzh and nnzj
- cons!, jac_structure!, jac_coord!, jprod, jtprod, jac_op
- hess_coord!, hprod!

## Debate
- Have different models or one with options ?
  The advantage of having options is the possibility to easily change the behavior
  of an NLPModels during the execution of an algorithms.

  However, it should be slower ?
- Is it an AbstractNLPModel or an AbstractADNLPModel?
