struct ZygoteADGradient <: ADBackend end
struct ZygoteADJacobian <: ADBackend
  nnzj::Int
end
struct ZygoteADHessian <: ADBackend
  nnzh::Int
end
struct ZygoteADJprod <: ADBackend end
struct ZygoteADJtprod <: ADBackend end

@init begin
  @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" begin
    function ZygoteADGradient(nvar::Integer, f, ncon::Integer = 0; kwargs...)
      return ZygoteADGradient()
    end
    function gradient(::ZygoteADGradient, f, x)
      g = Zygote.gradient(f, x)[1]
      return g === nothing ? zero(x) : g
    end
    function gradient!(::ZygoteADGradient, g, f, x)
      _g = Zygote.gradient(f, x)[1]
      g .= _g === nothing ? 0 : _g
    end

    function ZygoteADJacobian(nvar::Integer, f, ncon::Integer = 0; kwargs...)
      @assert nvar > 0
      nnzj = nvar * ncon
      return ZygoteADJacobian(nnzj)
    end
    function jacobian(::ZygoteADJacobian, f, x)
      return Zygote.jacobian(f, x)[1]
    end

    function ZygoteADHessian(nvar::Integer, f, ncon::Integer = 0; kwargs...)
      @assert nvar > 0
      nnzh = nvar * (nvar + 1) / 2
      return ZygoteADHessian(nnzh)
    end
    function hessian(b::ZygoteADHessian, f, x)
      return jacobian(
        ForwardDiffADJacobian(length(x), f, x0 = x),
        x -> gradient(ZygoteADGradient(), f, x),
        x,
      )
    end

    function ZygoteADJprod(nvar::Integer, f, ncon::Integer = 0; kwargs...)
      return ZygoteADJprod()
    end
    function Jprod(::ZygoteADJprod, f, x, v)
      return vec(Zygote.jacobian(t -> f(x + t * v), 0)[1])
    end

    function ZygoteADJtprod(nvar::Integer, f, ncon::Integer = 0; kwargs...)
      return ZygoteADJtprod()
    end
    function Jtprod(::ZygoteADJtprod, f, x, v)
      g = Zygote.gradient(x -> dot(f(x), v), x)[1]
      return g === nothing ? zero(x) : g
    end
  end
end
