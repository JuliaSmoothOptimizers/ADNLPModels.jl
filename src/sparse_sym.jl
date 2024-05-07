function compute_hessian_sparsity(f, nvar, c!, ncon)
  Symbolics.@variables xs[1:nvar]
  xsi = Symbolics.scalarize(xs)
  fun = f(xsi)
  if ncon > 0
    Symbolics.@variables ys[1:ncon]
    ysi = Symbolics.scalarize(ys)
    cx = similar(ysi)
    fun = fun + dot(c!(cx, xsi), ysi)
  end
  S = Symbolics.hessian_sparsity(fun, ncon == 0 ? xsi : [xsi; ysi]) # , full = false
  return S
end

function compute_jacobian_sparsity(c!, cx, x0)
  S = Symbolics.jacobian_sparsity(c!, cx, x0)
  return S
end

## ----- Symbolics Jacobian -----

struct SparseSymbolicsADJacobian{T} <: ADBackend
  nnzj::Int
  rows::Vector{Int}
  cols::Vector{Int}
  cfJ::T
end

function SparseSymbolicsADJacobian(nvar, f, ncon, c!; kwargs...)
  Symbolics.@variables xs[1:nvar] out[1:ncon]
  wi = Symbolics.scalarize(xs)
  wo = Symbolics.scalarize(out)
  fun = c!(wo, wi)
  J = Symbolics.jacobian_sparsity(c!, wo, wi)
  rows, cols, _ = findnz(J)
  vals = Symbolics.sparsejacobian_vals(fun, wi, rows, cols)
  nnzj = length(vals)
  # cfJ is a Tuple{Expr, Expr}, cfJ[2] is the in-place function
  # that we need to update a vector `vals` with the nonzeros of Jc(x).
  cfJ = Symbolics.build_function(vals, wi, expression = Val{false})
  SparseSymbolicsADJacobian(nnzj, rows, cols, cfJ[2])
end

function get_nln_nnzj(b::SparseSymbolicsADJacobian, nvar, ncon)
  b.nnzj
end

function NLPModels.jac_structure!(
  b::SparseSymbolicsADJacobian,
  nlp::ADModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  rows .= b.rows
  cols .= b.cols
  return rows, cols
end

function NLPModels.jac_coord!(
  b::SparseSymbolicsADJacobian,
  nlp::ADModel,
  x::AbstractVector,
  vals::AbstractVector,
)
  @eval $(b.cfJ)($vals, $x)
  return vals
end

function NLPModels.jac_structure_residual!(
  b::SparseSymbolicsADJacobian,
  nls::AbstractADNLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  rows .= b.rows
  cols .= b.cols
  return rows, cols
end

function NLPModels.jac_coord_residual!(
  b::SparseSymbolicsADJacobian,
  nls::AbstractADNLSModel,
  x::AbstractVector,
  vals::AbstractVector,
)
  @eval $(b.cfJ)($vals, $x)
  return vals
end

## ----- Symbolics Hessian -----

struct SparseSymbolicsADHessian{T, H} <: ADBackend
  nnzh::Int
  rows::Vector{Int}
  cols::Vector{Int}
  y::AbstractVector{T}
  cfH::H
end

function SparseSymbolicsADHessian(
  nvar,
  f,
  ncon,
  c!;
  x0::S = rand(nvar),
  kwargs...,
) where {S}
  Symbolics.@variables xs[1:nvar], μs
  xsi = Symbolics.scalarize(xs)
  fun = μs * f(xsi)
  Symbolics.@variables ys[1:ncon]
  ysi = Symbolics.scalarize(ys)
  if ncon > 0
    cx = similar(ysi)
    fun = fun + dot(c!(cx, xsi), ysi)
  end
  H = Symbolics.hessian_sparsity(fun, ncon == 0 ? xsi : [xsi; ysi], full = false)
  H = ncon == 0 ? H : H[1:nvar, 1:nvar]
  rows, cols, _ = findnz(H)
  vals = Symbolics.sparsehessian_vals(fun, xsi, rows, cols)
  nnzh = length(vals)
  # cfH is a Tuple{Expr, Expr}, cfH[2] is the in-place function
  # that we need to update a vector `vals` with the nonzeros of ∇²ℓ(x, y, μ).
  cfH = Symbolics.build_function(vals, xsi, ysi, μs, expression = Val{false})
  y = fill!(S(undef, ncon), 0)
  return SparseSymbolicsADHessian(nnzh, rows, cols, y, cfH[2])
end

function get_nln_nnzh(b::SparseSymbolicsADHessian, nvar)
  b.nnzh
end

function NLPModels.hess_structure!(
  b::SparseSymbolicsADHessian,
  nlp::ADModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  rows .= b.rows
  cols .= b.cols
  return rows, cols
end

function NLPModels.hess_coord!(
  b::SparseSymbolicsADHessian,
  nlp::ADModel,
  x::AbstractVector,
  y::AbstractVector,
  obj_weight::Real,
  vals::AbstractVector,
)
  @eval $(b.cfH)($vals, $x, $y, $obj_weight)
  return vals
end

function NLPModels.hess_coord!(
  b::SparseSymbolicsADHessian,
  nlp::ADModel,
  x::AbstractVector,
  obj_weight::Real,
  vals::AbstractVector,
)
  b.y .= 0
  @eval $(b.cfH)($vals, $x, $(b.y), $obj_weight)
  return vals
end

function NLPModels.hess_coord!(
  b::SparseSymbolicsADHessian,
  nlp::ADModel,
  x::AbstractVector,
  j::Integer,
  vals::AbstractVector{T},
) where {T}
  for (w, k) in enumerate(nlp.meta.nln)
    b.y[w] = k == j ? 1 : 0
  end
  obj_weight = zero(T)
  @eval $(b.cfH)($vals, $x, $(b.y), $obj_weight)
  return vals
end
