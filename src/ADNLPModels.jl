module ADNLPModels

# stdlib
using LinearAlgebra
# external
using ForwardDiff, ReverseDiff, Requires
# JSO
using NLPModels

include("ad.jl")
include("nlp.jl")
include("nls.jl")

export ForwardDiffAD, ZygoteAD, ReverseDiffAD

# We declare typical backends for continuity purpose and tests.
ForwardDiffAD(n, f, x0) = ForwardDiffAD(n, 0, f, x -> eltype(x0)[], x0)
ForwardDiffAD(n, ncon, f, x0) = ForwardDiffAD(n, ncon, f, x -> zeros(eltype(x0), ncon), x0)
function ForwardDiffAD(nvar, ncon, f, c, x0)
  gradient_backend = ADNLPModels.GradientForwardDiff(ForwardDiff.GradientConfig(f, x0))
  hprod_backend = ADNLPModels.HvForwardDiff(similar(x0, nvar))
  jprod_backend = ADNLPModels.JvForwardDiff()
  jtprod_backend = ADNLPModels.JtvForwardDiff()
  jacobian_backend = ADNLPModels.JacobianForwardDiff()
  hessian_backend = ADNLPModels.HessianForwardDiff()
  ADModelBackend(nvar, ncon, f, c, x0, gradient_backend = gradient_backend, hprod_backend = hprod_backend, jprod_backend = jprod_backend, jtprod_backend = jtprod_backend, jacobian_backend = jacobian_backend, hessian_backend = hessian_backend)
end

ReverseDiffAD(n, f, x0) = ReverseDiffAD(n, 0, f, x -> eltype(x0)[], x0)
ReverseDiffAD(n, ncon, f, x0) = ReverseDiffAD(n, ncon, f, x -> zeros(eltype(x0), ncon), x0)
function ReverseDiffAD(nvar, ncon, f, c, x0)
  gradient_backend = ADNLPModels.GradientReverseDiff(ReverseDiff.GradientTape(f, x0, ReverseDiff.GradientConfig(x0)))  
  hprod_backend = ADNLPModels.HvReverseDiff(similar(x0, nvar))
  jprod_backend = ADNLPModels.JvReverseDiff()
  jtprod_backend = ADNLPModels.JtvReverseDiff()
  jacobian_backend = ADNLPModels.JacobianReverseDiff()
  hessian_backend = ADNLPModels.HessianReverseDiff()
  ADModelBackend(nvar, ncon, f, c, x0, gradient_backend = gradient_backend, hprod_backend = hprod_backend, jprod_backend = jprod_backend, jtprod_backend = jtprod_backend, jacobian_backend = jacobian_backend, hessian_backend = hessian_backend)
end

@init begin
  @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" begin
    ZygoteAD(n, f, x0) = ZygoteAD(n, 0, f, x -> eltype(x0)[], x0)
    ZygoteAD(n, ncon, f, x0) = ZygoteAD(n, ncon, f, x -> zeros(eltype(x0), ncon), x0)
    function ZygoteAD(nvar, ncon, f, c, x0)
      gradient_backend = ADNLPModels.GradientZygote()
      hprod_backend = ADNLPModels.HvZygote()
      jprod_backend = ADNLPModels.JvZygote()
      jtprod_backend = ADNLPModels.JtvZygote()
      jacobian_backend = ADNLPModels.JacobianZygote()
      hessian_backend = ADNLPModels.HessianZygote()
      ADModelBackend(nvar, ncon, f, c, x0, gradient_backend = gradient_backend, hprod_backend = hprod_backend, jprod_backend = jprod_backend, jtprod_backend = jtprod_backend, jacobian_backend = jacobian_backend, hessian_backend = hessian_backend)
    end
  end
end

end # module
