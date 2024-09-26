using CUDA, LinearAlgebra, SparseArrays, Test
using ADNLPModels, NLPModels, NLPModelsTest

for problem in NLPModelsTest.nlp_problems ∪ ["GENROSE"]
  include("../nlp/problems/$(lowercase(problem)).jl")
end
for problem in NLPModelsTest.nls_problems
  include("../nls/problems/$(lowercase(problem)).jl")
end

@test CUDA.functional()

@testset "Checking NLPModelsTest (NLP) tests with $backend - GPU multiple precision" for backend in
                                                                                         keys(
  ADNLPModels.predefined_backend,
)
  @testset "Checking GPU multiple precision on problem $problem" for problem in
                                                                     NLPModelsTest.nlp_problems
    nlp_from_T = eval(Meta.parse(lowercase(problem) * "_autodiff"))
    CUDA.allowscalar() do
      # sparse Jacobian/Hessian doesn't work here
      multiple_precision_nlp_array(
        T -> nlp_from_T(
          T;
          jacobian_backend = ADNLPModels.ForwardDiffADJacobian,
          hessian_backend = ADNLPModels.ForwardDiffADHessian,
        ),
        CuArray,
        exclude = [jth_hprod, hprod, jprod],
        linear_api = true,
      )
    end
  end
end

@testset "Checking NLPModelsTest (NLS) tests with $backend - GPU multiple precision" for backend in
                                                                                         keys(
  ADNLPModels.predefined_backend,
)
  @testset "Checking GPU multiple precision on problem $problem" for problem in
                                                                     NLPModelsTest.nls_problems
    nls_from_T = eval(Meta.parse(lowercase(problem) * "_autodiff"))
    CUDA.allowscalar() do
      # sparse Jacobian/Hessian doesn't work here
      multiple_precision_nls_array(
        T -> nls_from_T(
          T;
          jacobian_backend = ADNLPModels.ForwardDiffADJacobian,
          hessian_backend = ADNLPModels.ForwardDiffADHessian,
          jacobian_residual_backend = ADNLPModels.ForwardDiffADJacobian,
          hessian_residual_backend = ADNLPModels.ForwardDiffADHessian,
        ),
        CuArray,
        exclude = [jprod, jprod_residual, hprod_residual],
        linear_api = true,
      )
    end
  end
end
