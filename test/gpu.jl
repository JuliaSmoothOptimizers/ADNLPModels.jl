using CUDA, LinearAlgebra, SparseArrays, Test
using ADNLPModels, ManualNLPModels, NLPModels, NLPModelsModifiers, NLPModelsTest

@test CUDA.functional()

@testset "Check GPU multiple precision" begin
  @testset "NLP" begin
    CUDA.allowscalar() do
      # sparse Jacobian/Hessian doesn't work here
      multiple_precision_nlp_array(T -> nlp_from_T(T; jacobian_backend = ADNLPModels.ForwardDiffADJacobian, hessian_backend = ADNLPModels.ForwardDiffADHessian), CuArray, exclude = [jth_hprod], linear_api = true)
    end
  end

  @testset "NLS" begin
    CUDA.allowscalar() do
      # sparse Jacobian/Hessian doesn't work here
      multiple_precision_nls_array(T -> nls_from_T(T; jacobian_backend = ADNLPModels.ForwardDiffADJacobian, hessian_backend = ADNLPModels.ForwardDiffADHessian, jacobian_residual_backend = ADNLPModels.ForwardDiffADJacobian, hessian_residual_backend = ADNLPModels.ForwardDiffADHessian), CuArray, exclude = [], linear_api = true)
    end
  end
end
