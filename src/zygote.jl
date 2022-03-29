struct ZygoteAD <: ADBackend
  nnzh::Int
  nnzj::Int
end

@init begin
  @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" begin
    function ZygoteAD(nvar::Integer, f, ncon::Integer = 0; kwargs...)
      @assert nvar > 0
      nnzh = nvar * (nvar + 1) / 2
      nnzj = nvar * ncon
      return ZygoteAD(nnzh, nnzj)
    end

    function gradient(::ZygoteAD, f, x)
      g = Zygote.gradient(f, x)[1]
      return g === nothing ? zero(x) : g
    end
    function gradient!(::ZygoteAD, g, f, x)
      _g = Zygote.gradient(f, x)[1]
      g .= _g === nothing ? 0 : _g
    end
    function jacobian(::ZygoteAD, f, x)
      return Zygote.jacobian(f, x)[1]
    end
    function hessian(b::ZygoteAD, f, x)
      return jacobian(ForwardDiffAD(length(x), f, x0 = x), x -> gradient(b, f, x), x)
    end
    function Jprod(::ZygoteAD, f, x, v)
      return vec(Zygote.jacobian(t -> f(x + t * v), 0)[1])
    end
    function Jtprod(::ZygoteAD, f, x, v)
      g = Zygote.gradient(x -> dot(f(x), v), x)[1]
      return g === nothing ? zero(x) : g
    end
  end
end
