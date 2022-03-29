struct ReverseDiffAD{T} <: ADBackend
  nnzh::Int
  nnzj::Int
  cfg::T
end

@init begin
  @require ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267" begin
    function ReverseDiffAD(
      nvar::Integer,
      f,
      ncon::Integer = 0;
      x0::AbstractVector = rand(nvar),
      kwargs...,
    )
      @assert nvar > 0
      @lencheck nvar x0
      nnzh = nvar * (nvar + 1) / 2
      nnzj = nvar * ncon
      f_tape = ReverseDiff.GradientTape(f, x0)
      cfg = ReverseDiff.compile(f_tape)
      return ReverseDiffAD{typeof(cfg)}(nnzh, nnzj, cfg)
    end

    gradient(adbackend::ReverseDiffAD, f, x) = ReverseDiff.gradient(f, x, adbackend.cfg)
    function gradient!(adbackend::ReverseDiffAD, g, f, x)
      return ReverseDiff.gradient!(g, adbackend.cfg, x)
    end
    jacobian(::ReverseDiffAD, f, x) = ReverseDiff.jacobian(f, x)
    hessian(::ReverseDiffAD, f, x) = ReverseDiff.hessian(f, x)
    function Jprod(::ReverseDiffAD, f, x, v)
      return vec(ReverseDiff.jacobian(t -> f(x + t[1] * v), [0.0]))
    end
    function Jtprod(::ReverseDiffAD, f, x, v)
      return ReverseDiff.gradient(x -> dot(f(x), v), x)
    end
    function Hvprod(::ReverseDiffAD, f, x, v)
      return ForwardDiff.derivative(t -> ReverseDiff.gradient(f, x + t * v), 0)
    end
  end
end
