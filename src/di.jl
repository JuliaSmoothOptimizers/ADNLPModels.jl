for (ADGradient, fbackend) in ((:EnzymeADGradient              , :AutoEnzyme              ),
                               (:ZygoteADGradient              , :AutoZygote              ),
                             # (:ForwardDiffADGradient         , :AutoForwardDiff         ),
                             # (:ReverseDiffADGradient         , :AutoReverseDiff         ),
                               (:MooncakeADGradient            , :AutoMooncake            ),
                               (:DiffractorADGradient          , :AutoDiffractor          ),
                               (:TrackerADGradient             , :AutoTracker             ),
                               (:SymbolicsADGradient           , :AutoSymbolics           ),
                               (:ChainRulesADGradient          , :AutoChainRules          ),
                               (:FastDifferentiationADGradient , :AutoFastDifferentiation ),
                               (:FiniteDiffADGradient          , :AutoFiniteDiff          ),
                               (:FiniteDifferencesADGradient   , :AutoFiniteDifferences   ),
                               (:PolyesterForwardDiffADGradient, :AutoPolyesterForwardDiff))
  @eval begin

    struct $ADGradient{B, E} <: ADBackend
      backend::B
      prep::E
    end

    function $ADGradient(
      nvar::Integer,
      f,
      ncon::Integer = 0,
      c::Function = (args...) -> [];
      x0::AbstractVector = rand(nvar),
      kwargs...,
    )
      backend = $fbackend()
      prep = DifferentiationInterface.prepare_gradient(f, backend, x0)
      return $ADGradient(backend, prep)
    end

    function gradient(b::$ADGradient, f, x)
      g = DifferentiationInterface.gradient(f, b.prep, b.backend, x)
      return g
    end

    function gradient!(b::$ADGradient, g, f, x)
      DifferentiationInterface.gradient!(f, g, b.prep, b.backend, x)
      return g
    end

  end
end

for (ADJprod, fbackend) in ((:EnzymeADJprod              , :AutoEnzyme              ),
                            (:ZygoteADJprod              , :AutoZygote              ),
                          # (:ForwardDiffADJprod         , :AutoForwardDiff         ),
                          # (:ReverseDiffADJprod         , :AutoReverseDiff         ),
                            (:MooncakeADJprod            , :AutoMooncake            ),
                            (:DiffractorADJprod          , :AutoDiffractor          ),
                            (:TrackerADJprod             , :AutoTracker             ),
                            (:SymbolicsADJprod           , :AutoSymbolics           ),
                            (:ChainRulesADJprod          , :AutoChainRules          ),
                            (:FastDifferentiationADJprod , :AutoFastDifferentiation ),
                            (:FiniteDiffADJprod          , :AutoFiniteDiff          ),
                            (:FiniteDifferencesADJprod   , :AutoFiniteDifferences   ),
                            (:PolyesterForwardDiffADJprod, :AutoPolyesterForwardDiff))
  @eval begin

    struct $ADJprod{B, E} <: ADBackend
      backend::B
      prep::E
    end

    function $ADJprod(
      nvar::Integer,
      f,
      ncon::Integer = 0,
      c::Function = (args...) -> [];
      x0::AbstractVector = rand(nvar),
      kwargs...,
    )
      backend = $fbackend()
      dy = similar(x0, ncon)
      dx = similar(x0, nvar)
      prep = DifferentiationInterface.prepare_pushforward(c, dy, backend, x0, dx)
      return $ADJprod(backend, prep)
    end

    function Jprod!(b::$ADJprod, Jv, c, x, v, ::Val)
      DifferentiationInterface.pushforward!(c, Jv, b.prep, b.backend, x, v)
      return Jv
    end

  end
end

for (ADJtprod, fbackend) in ((:EnzymeADJtprod              , :AutoEnzyme              ),
                             (:ZygoteADJtprod              , :AutoZygote              ),
                           # (:ForwardDiffADJtprod         , :AutoForwardDiff         ),
                           # (:ReverseDiffADJtprod         , :AutoReverseDiff         ),
                             (:MooncakeADJtprod            , :AutoMooncake            ),
                             (:DiffractorADJtprod          , :AutoDiffractor          ),
                             (:TrackerADJtprod             , :AutoTracker             ),
                             (:SymbolicsADJtprod           , :AutoSymbolics           ),
                             (:ChainRulesADJtprod          , :AutoChainRules          ),
                             (:FastDifferentiationADJtprod , :AutoFastDifferentiation ),
                             (:FiniteDiffADJtprod          , :AutoFiniteDiff          ),
                             (:FiniteDifferencesADJtprod   , :AutoFiniteDifferences   ),
                             (:PolyesterForwardDiffADJtprod, :AutoPolyesterForwardDiff))
  @eval begin

    struct $ADJtprod{B, E} <: ADBackend
      backend::B
      prep::E
    end

    function $ADJtprod(
      nvar::Integer,
      f,
      ncon::Integer = 0,
      c::Function = (args...) -> [];
      x0::AbstractVector = rand(nvar),
      kwargs...,
    )
      backend = $fbackend()
      dx = similar(x0, nvar)
      dy = similar(x0, ncon)
      prep = DifferentiationInterface.prepare_pullback(c, dx, backend, x0, dy)
      return $ADJtprod(backend, prep)
    end

    function Jtprod!(b::$ADJtprod, Jtv, c, x, v, ::Val)
      DifferentiationInterface.pullback!(c, Jtv, b.prep, b.backend, x, v)
      return Jtv
    end

  end
end

for (ADJacobian, fbackend) in ((:EnzymeADJacobian              , :AutoEnzyme              ),
                               (:ZygoteADJacobian              , :AutoZygote              ),
                             # (:ForwardDiffADJacobian         , :AutoForwardDiff         ),
                             # (:ReverseDiffADJacobian         , :AutoReverseDiff         ),
                               (:MooncakeADJacobian            , :AutoMooncake            ),
                               (:DiffractorADJacobian          , :AutoDiffractor          ),
                               (:TrackerADJacobian             , :AutoTracker             ),
                               (:SymbolicsADJacobian           , :AutoSymbolics           ),
                               (:ChainRulesADJacobian          , :AutoChainRules          ),
                               (:FastDifferentiationADJacobian , :AutoFastDifferentiation ),
                               (:FiniteDiffADJacobian          , :AutoFiniteDiff          ),
                               (:FiniteDifferencesADJacobian   , :AutoFiniteDifferences   ),
                               (:PolyesterForwardDiffADJacobian, :AutoPolyesterForwardDiff))
  @eval begin

    struct $ADJacobian{B, E} <: ADBackend
      backend::B
      prep::E
    end

    function $ADJacobian(
      nvar::Integer,
      f,
      ncon::Integer = 0,
      c::Function = (args...) -> [];
      x0::AbstractVector = rand(nvar),
      kwargs...,
    )
      backend = $fbackend()
      y = similar(x0, ncon)
      prep = DifferentiationInterface.prepare_jacobian(c, y, backend, x0)
      return $ADJacobian(backend, prep)
    end

    function jacobian(b::$ADJacobian, c, x)
      J = DifferentiationInterface.jacobian(c, b.prep, b.backend, x)
      return J
    end

  end
end

for (ADHvprod, fbackend) in ((:EnzymeADHvprod              , :AutoEnzyme              ),
                             (:ZygoteADHvprod              , :AutoZygote              ),
                           # (:ForwardDiffADHvprod         , :AutoForwardDiff         ),
                           # (:ReverseDiffADHvprod         , :AutoReverseDiff         ),
                             (:MooncakeADHvprod            , :AutoMooncake            ),
                             (:DiffractorADHvprod          , :AutoDiffractor          ),
                             (:TrackerADHvprod             , :AutoTracker             ),
                             (:SymbolicsADHvprod           , :AutoSymbolics           ),
                             (:ChainRulesADHvprod          , :AutoChainRules          ),
                             (:FastDifferentiationADHvprod , :AutoFastDifferentiation ),
                             (:FiniteDiffADHvprod          , :AutoFiniteDiff          ),
                             (:FiniteDifferencesADHvprod   , :AutoFiniteDifferences   ),
                             (:PolyesterForwardDiffADHvprod, :AutoPolyesterForwardDiff))
  @eval begin

    struct $ADHvprod{B, E} <: ADBackend
      backend::B
      prep::E
    end

    function $ADHvprod(
      nvar::Integer,
      f,
      ncon::Integer = 0,
      c::Function = (args...) -> [];
      x0::AbstractVector = rand(nvar),
      kwargs...,
    )
      backend = $fbackend()
      tx = similar(x0)
      prep = DifferentiationInterface.prepare_hvp(f, backend, x0, tx)
      return $ADHvprod(backend, prep)
    end

    function Hvprod!(b::$ADHvprod, Hv, f, x, v, ::Val)
      DifferentiationInterface.hvp!(f, Hv, b.prep, b.backend, x, v)
      return Hv
    end

  end
end

for (ADHessian, fbackend) in ((:EnzymeADHessian              , :AutoEnzyme              ),
                              (:ZygoteADHessian              , :AutoZygote              ),
                            # (:ForwardDiffADHessian         , :AutoForwardDiff         ),
                            # (:ReverseDiffADHessian         , :AutoReverseDiff         ),
                              (:MooncakeADHessian            , :AutoMooncake            ),
                              (:DiffractorADHessian          , :AutoDiffractor          ),
                              (:TrackerADHessian             , :AutoTracker             ),
                              (:SymbolicsADHessian           , :AutoSymbolics           ),
                              (:ChainRulesADHessian          , :AutoChainRules          ),
                              (:FastDifferentiationADHessian , :AutoFastDifferentiation ),
                              (:FiniteDiffADHessian          , :AutoFiniteDiff          ),
                              (:FiniteDifferencesADHessian   , :AutoFiniteDifferences   ),
                              (:PolyesterForwardDiffADHessian, :AutoPolyesterForwardDiff))
  @eval begin

    struct $ADHessian{B, E} <: ADBackend
      backend::B
      prep::E
    end

    function $ADHessian(
      nvar::Integer,
      f,
      ncon::Integer = 0,
      c::Function = (args...) -> [];
      x0::AbstractVector = rand(nvar),
      kwargs...,
    )
      backend = $fbackend()
      prep = DifferentiationInterface.prepare_hessian(f, backend, x0)
      return $ADHessian(backend, prep)
    end

    function hessian(b::$ADHessian, f, x)
      H = DifferentiationInterface.hessian(f, b.prep, b.backend, x)
      return H
    end

  end
end
