# ADNLPModels

## TODO
### Tests
- get tests from NLPModels
- new test problems with funny structure
### Benchmark
- ff
### Code (1st goal is for unconstrained)
- compute nnzh
- grad!, hess_structure!, hess_coord!, hprod!
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