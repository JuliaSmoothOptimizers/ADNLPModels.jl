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
  :gradient_backend => ReverseDiffADGradient, # EnzymeADGradient
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

predefined_backend = Dict(:default => default_backend, :optimized => optimized, :generic => generic)

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
