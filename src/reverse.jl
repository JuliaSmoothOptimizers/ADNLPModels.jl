struct ReverseDiffADGradient <: ADBackend
  cfg
end
struct ReverseDiffADJacobian <: ADBackend 
  nnzj::Int
end
struct ReverseDiffADHessian <: ADBackend
  nnzh::Int
end
struct ReverseDiffADJprod <: ADBackend end
struct ReverseDiffADJtprod <: ADBackend end
struct ReverseDiffADHvprod <: ADBackend end

@init begin
  @require ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267" begin

    function ReverseDiffADGradient(
      nvar::Integer,
      f,
      ncon::Integer = 0;
      x0::AbstractVector = rand(nvar),
      kwargs...,
    )
      @assert nvar > 0
      @lencheck nvar x0
      f_tape = ReverseDiff.GradientTape(f, x0)
      cfg = ReverseDiff.compile(f_tape)
      return ReverseDiffADGradient(cfg)
    end
    gradient(adbackend::ReverseDiffADGradient, f, x) = ReverseDiff.gradient(f, x, adbackend.cfg)
    function gradient!(adbackend::ReverseDiffADGradient, g, f, x)
      return ReverseDiff.gradient!(g, adbackend.cfg, x)
    end
    
    function ReverseDiffADJacobian(
      nvar::Integer,
      f,
      ncon::Integer = 0;
      kwargs...,
    )
      @assert nvar > 0
      nnzj = nvar * ncon
      return ReverseDiffADJacobian(nnzj)
    end
    jacobian(::ReverseDiffADJacobian, f, x) = ReverseDiff.jacobian(f, x)
    
    function ReverseDiffADHessian(
      nvar::Integer,
      f,
      ncon::Integer = 0;
      kwargs...,
    )
      @assert nvar > 0
      nnzh = nvar * (nvar + 1) / 2
      return ReverseDiffADHessian(nnzh)
    end
    hessian(::ReverseDiffADHessian, f, x) = ReverseDiff.hessian(f, x)
    
    function ReverseDiffADJprod(
      nvar::Integer,
      f,
      ncon::Integer = 0;
      kwargs...,
    )
      return ReverseDiffADJprod()
    end
    function Jprod(::ReverseDiffADJprod, f, x, v)
      return vec(ReverseDiff.jacobian(t -> f(x + t[1] * v), [0.0]))
    end
    
    function ReverseDiffADJtprod(
      nvar::Integer,
      f,
      ncon::Integer = 0;
      kwargs...,
    )
      return ReverseDiffADJtprod()
    end
    function Jtprod(::ReverseDiffADJtprod, f, x, v)
      return ReverseDiff.gradient(x -> dot(f(x), v), x)
    end
    
    function ReverseDiffADHvprod(
      nvar::Integer,
      f,
      ncon::Integer = 0;
      kwargs...,
    )
      return ReverseDiffADHvprod()
    end
    function Hvprod(::ReverseDiffADHvprod, f, x, v)
      return ForwardDiff.derivative(t -> ReverseDiff.gradient(f, x + t * v), 0)
    end
  end
end
