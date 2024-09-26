default_backend = Dict(
  :gradient_backend => ForwardDiffADGradient,
  :hprod_backend => ForwardDiffADHvprod,
  :jprod_backend => ForwardDiffADJprod,
  :jtprod_backend => ForwardDiffADJtprod,
  :jacobian_backend => SparseADJacobian,
  :hessian_backend => SparseADHessian,
  :ghjvprod_backend => ForwardDiffADGHjvprod,
  :hprod_residual_backend => ForwardDiffADHvprod,
  :jprod_residual_backend => ForwardDiffADJprod,
  :jtprod_residual_backend => ForwardDiffADJtprod,
  :jacobian_residual_backend => SparseADJacobian,
  :hessian_residual_backend => SparseADHessian,
)

optimized = Dict(
  :gradient_backend => ReverseDiffADGradient,
  :hprod_backend => ReverseDiffADHvprod,
  :jprod_backend => ForwardDiffADJprod,
  :jtprod_backend => ReverseDiffADJtprod,
  :jacobian_backend => SparseADJacobian,
  :hessian_backend => SparseReverseADHessian,
  :ghjvprod_backend => ForwardDiffADGHjvprod,
  :hprod_residual_backend => ReverseDiffADHvprod,
  :jprod_residual_backend => ForwardDiffADJprod,
  :jtprod_residual_backend => ReverseDiffADJtprod,
  :jacobian_residual_backend => SparseADJacobian,
  :hessian_residual_backend => SparseReverseADHessian,
)

generic = Dict(
  :gradient_backend => GenericForwardDiffADGradient,
  :hprod_backend => GenericForwardDiffADHvprod,
  :jprod_backend => GenericForwardDiffADJprod,
  :jtprod_backend => GenericForwardDiffADJtprod,
  :jacobian_backend => ForwardDiffADJacobian,
  :hessian_backend => ForwardDiffADHessian,
  :ghjvprod_backend => ForwardDiffADGHjvprod,
  :hprod_residual_backend => GenericForwardDiffADHvprod,
  :jprod_residual_backend => GenericForwardDiffADJprod,
  :jtprod_residual_backend => GenericForwardDiffADJtprod,
  :jacobian_residual_backend => ForwardDiffADJacobian,
  :hessian_residual_backend => ForwardDiffADHessian,
)

ForwardDiff_backend = Dict(
  :gradient_backend => ForwardDiffADGradient,
  :jprod_backend => ForwardDiffADJprod,
  :jtprod_backend => ForwardDiffADJtprod,
  :hprod_backend => ForwardDiffADHvprod,
  :jacobian_backend => ForwardDiffADJacobian,
  :hessian_backend => ForwardDiffADHessian,
  :ghjvprod_backend => EmptyADbackend,
  :jprod_residual_backend => ForwardDiffADJprod,
  :jtprod_residual_backend => ForwardDiffADJtprod,
  :hprod_residual_backend => ForwardDiffADHvprod,
  :jacobian_residual_backend => ForwardDiffADJacobian,
  :hessian_residual_backend => ForwardDiffADHessian
)

ReverseDiff_backend = Dict(
  :gradient_backend => ReverseDiffADGradient,
  :jprod_backend => ReverseDiffADJprod,
  :jtprod_backend => ReverseDiffADJtprod,
  :hprod_backend => ReverseDiffADHvprod,
  :jacobian_backend => ReverseDiffADJacobian,
  :hessian_backend => ReverseDiffADHessian,
  :ghjvprod_backend => EmptyADbackend,
  :jprod_residual_backend => ReverseDiffADJprod,
  :jtprod_residual_backend => ReverseDiffADJtprod,
  :hprod_residual_backend => ReverseDiffADHvprod,
  :jacobian_residual_backend => ReverseDiffADJacobian,
  :hessian_residual_backend => ReverseDiffADHessian
)

Enzyme_backend = Dict(
  :gradient_backend => EnzymeADGradient,
  :jprod_backend => EnzymeADJprod,
  :jtprod_backend => EnzymeADJtprod,
  :hprod_backend => EnzymeADHvprod,
  :jacobian_backend => EnzymeADJacobian,
  :hessian_backend => EnzymeADHessian,
  :ghjvprod_backend => EmptyADbackend,
  :jprod_residual_backend => EnzymeADJprod,
  :jtprod_residual_backend => EnzymeADJtprod,
  :hprod_residual_backend => EnzymeADHvprod,
  :jacobian_residual_backend => EnzymeADJacobian,
  :hessian_residual_backend => EnzymeADHessian
)

Zygote_backend = Dict(
  :gradient_backend => ZygoteADGradient,
  :jprod_backend => ZygoteADJprod,
  :jtprod_backend => ZygoteADJtprod,
  :hprod_backend => ZygoteADHvprod,
  :jacobian_backend => ZygoteADJacobian,
  :hessian_backend => ZygoteADHessian,
  :ghjvprod_backend => EmptyADbackend,
  :jprod_residual_backend => ZygoteADJprod,
  :jtprod_residual_backend => ZygoteADJtprod,
  :hprod_residual_backend => ZygoteADHvprod,
  :jacobian_residual_backend => ZygoteADJacobian,
  :hessian_residual_backend => ZygoteADHessian
)

Mooncake_backend = Dict(
  :gradient_backend => MooncakeADGradient,
  :jprod_backend => MooncakeADJprod,
  :jtprod_backend => MooncakeADJtprod,
  :hprod_backend => MooncakeADHvprod,
  :jacobian_backend => MooncakeADJacobian,
  :hessian_backend => MooncakeADHessian,
  :ghjvprod_backend => EmptyADbackend,
  :jprod_residual_backend => MooncakeADJprod,
  :jtprod_residual_backend => MooncakeADJtprod,
  :hprod_residual_backend => MooncakeADHvprod,
  :jacobian_residual_backend => MooncakeADJacobian,
  :hessian_residual_backend => MooncakeADHessian
)

Diffractor_backend = Dict(
  :gradient_backend => DiffractorADGradient,
  :jprod_backend => DiffractorADJprod,
  :jtprod_backend => DiffractorADJtprod,
  :hprod_backend => DiffractorADHvprod,
  :jacobian_backend => DiffractorADJacobian,
  :hessian_backend => DiffractorADHessian,
  :ghjvprod_backend => EmptyADbackend,
  :jprod_residual_backend => DiffractorADJprod,
  :jtprod_residual_backend => DiffractorADJtprod,
  :hprod_residual_backend => DiffractorADHvprod,
  :jacobian_residual_backend => DiffractorADJacobian,
  :hessian_residual_backend => DiffractorADHessian
)

Tracker_backend = Dict(
  :gradient_backend => TrackerADGradient,
  :jprod_backend => TrackerADJprod,
  :jtprod_backend => TrackerADJtprod,
  :hprod_backend => TrackerADHvprod,
  :jacobian_backend => TrackerADJacobian,
  :hessian_backend => TrackerADHessian,
  :ghjvprod_backend => EmptyADbackend,
  :jprod_residual_backend => TrackerADJprod,
  :jtprod_residual_backend => TrackerADJtprod,
  :hprod_residual_backend => TrackerADHvprod,
  :jacobian_residual_backend => TrackerADJacobian,
  :hessian_residual_backend => TrackerADHessian
)

Symbolics_backend = Dict(
  :gradient_backend => SymbolicsADGradient,
  :jprod_backend => SymbolicsADJprod,
  :jtprod_backend => SymbolicsADJtprod,
  :hprod_backend => SymbolicsADHvprod,
  :jacobian_backend => SymbolicsADJacobian,
  :hessian_backend => SymbolicsADHessian,
  :ghjvprod_backend => EmptyADbackend,
  :jprod_residual_backend => SymbolicsADJprod,
  :jtprod_residual_backend => SymbolicsADJtprod,
  :hprod_residual_backend => SymbolicsADHvprod,
  :jacobian_residual_backend => SymbolicsADJacobian,
  :hessian_residual_backend => SymbolicsADHessian
)

ChainRules_backend = Dict(
  :gradient_backend => ChainRulesADGradient,
  :jprod_backend => ChainRulesADJprod,
  :jtprod_backend => ChainRulesADJtprod,
  :hprod_backend => ChainRulesADHvprod,
  :jacobian_backend => ChainRulesADJacobian,
  :hessian_backend => ChainRulesADHessian,
  :ghjvprod_backend => EmptyADbackend,
  :jprod_residual_backend => ChainRulesADJprod,
  :jtprod_residual_backend => ChainRulesADJtprod,
  :hprod_residual_backend => ChainRulesADHvprod,
  :jacobian_residual_backend => ChainRulesADJacobian,
  :hessian_residual_backend => ChainRulesADHessian
)

FastDifferentiation_backend = Dict(
  :gradient_backend => FastDifferentiationADGradient,
  :jprod_backend => FastDifferentiationADJprod,
  :jtprod_backend => FastDifferentiationADJtprod,
  :hprod_backend => FastDifferentiationADHvprod,
  :jacobian_backend => FastDifferentiationADJacobian,
  :hessian_backend => FastDifferentiationADHessian,
  :ghjvprod_backend => EmptyADbackend,
  :jprod_residual_backend => FastDifferentiationADJprod,
  :jtprod_residual_backend => FastDifferentiationADJtprod,
  :hprod_residual_backend => FastDifferentiationADHvprod,
  :jacobian_residual_backend => FastDifferentiationADJacobian,
  :hessian_residual_backend => FastDifferentiationADHessian
)

FiniteDiff_backend = Dict(
  :gradient_backend => FiniteDiffADGradient,
  :jprod_backend => FiniteDiffADJprod,
  :jtprod_backend => FiniteDiffADJtprod,
  :hprod_backend => FiniteDiffADHvprod,
  :jacobian_backend => FiniteDiffADJacobian,
  :hessian_backend => FiniteDiffADHessian,
  :ghjvprod_backend => EmptyADbackend,
  :jprod_residual_backend => FiniteDiffADJprod,
  :jtprod_residual_backend => FiniteDiffADJtprod,
  :hprod_residual_backend => FiniteDiffADHvprod,
  :jacobian_residual_backend => FiniteDiffADJacobian,
  :hessian_residual_backend => FiniteDiffADHessian
)

FiniteDifferences_backend = Dict(
  :gradient_backend => FiniteDifferencesADGradient,
  :jprod_backend => FiniteDifferencesADJprod,
  :jtprod_backend => FiniteDifferencesADJtprod,
  :hprod_backend => FiniteDifferencesADHvprod,
  :jacobian_backend => FiniteDifferencesADJacobian,
  :hessian_backend => FiniteDifferencesADHessian,
  :ghjvprod_backend => EmptyADbackend,
  :jprod_residual_backend => FiniteDifferencesADJprod,
  :jtprod_residual_backend => FiniteDifferencesADJtprod,
  :hprod_residual_backend => FiniteDifferencesADHvprod,
  :jacobian_residual_backend => FiniteDifferencesADJacobian,
  :hessian_residual_backend => FiniteDifferencesADHessian
)

PolyesterForwardDiff_backend = Dict(
  :gradient_backend => PolyesterForwardDiffADGradient,
  :jprod_backend => PolyesterForwardDiffADJprod,
  :jtprod_backend => PolyesterForwardDiffADJtprod,
  :hprod_backend => PolyesterForwardDiffADHvprod,
  :jacobian_backend => PolyesterForwardDiffADJacobian,
  :hessian_backend => PolyesterForwardDiffADHessian,
  :ghjvprod_backend => EmptyADbackend,
  :jprod_residual_backend => PolyesterForwardDiffADJprod,
  :jtprod_residual_backend => PolyesterForwardDiffADJtprod,
  :hprod_residual_backend => PolyesterForwardDiffADHvprod,
  :jacobian_residual_backend => PolyesterForwardDiffADJacobian,
  :hessian_residual_backend => PolyesterForwardDiffADHessian
)

predefined_backend = Dict(:default => default_backend,
                          :optimized => optimized,
                          :generic => generic,
                          :ForwardDiff => ForwardDiff_backend,
                          :ReverseDiff => ReverseDiff_backend,
                          :Enzyme => Enzyme_backend,
                          :Zygote => Zygote_backend,
                          :Mooncake => Mooncake_backend,
                          :Diffractor => Diffractor_backend,
                          :Tracker => Tracker_backend,
                          :Symbolics => Symbolics_backend,
                          :ChainRules => ChainRules_backend,
                          :FastDifferentiation => FastDifferentiation_backend,
                          :FiniteDiff => FiniteDiff_backend,
                          :FiniteDifferences => FiniteDifferences_backend,
                          :PolyesterForwardDiff => PolyesterForwardDiff_backend)

"""
    get_default_backend(meth::Symbol, backend::Symbol; kwargs...)
    get_default_backend(::Val{::Symbol}, backend; kwargs...)

Return a type `<:ADBackend` that corresponds to the default `backend` use for the method `meth`.
See `keys(ADNLPModels.predefined_backend)` for a list of possible backends.

The following keyword arguments are accepted:
- `matrix_free::Bool`: If `true`, this returns an `EmptyADbackend` for methods that handle matrices, e.g. `:hessian_backend`.

"""
function get_default_backend(meth::Symbol, args...; kwargs...)
  return get_default_backend(Val(meth), args...; kwargs...)
end

function get_default_backend(::Val{sym}, backend, args...; kwargs...) where {sym}
  return predefined_backend[backend][sym]
end

function get_default_backend(::Val{:jacobian_backend}, backend, matrix_free::Bool = false)
  return matrix_free ? EmptyADbackend : predefined_backend[backend][:jacobian_backend]
end
function get_default_backend(::Val{:hessian_backend}, backend, matrix_free::Bool = false)
  return matrix_free ? EmptyADbackend : predefined_backend[backend][:hessian_backend]
end
function get_default_backend(::Val{:jacobian_residual_backend}, backend, matrix_free::Bool = false)
  return matrix_free ? EmptyADbackend : predefined_backend[backend][:jacobian_residual_backend]
end
function get_default_backend(::Val{:hessian_residual_backend}, backend, matrix_free::Bool = false)
  return matrix_free ? EmptyADbackend : predefined_backend[backend][:hessian_residual_backend]
end
