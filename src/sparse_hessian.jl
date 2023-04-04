## ----- Symbolics -----

struct SparseADHessian{T} <: ADBackend
  nnzh::Int
  rows::Vector{Int}
  cols::Vector{Int}
  cfH::T
end

function SparseADHessian(nvar, f, ncon, c!; kwargs...)
  @variables xs[1:nvar], μs
  xsi = Symbolics.scalarize(xs)
  fun = μs * f(xsi)
  if ncon > 0
    @variables ys[1:ncon]
    ysi = Symbolics.scalarize(ys)
    cx = similar(ysi)
    fun = fun + dot(c!(cx, xsi), ysi)
  end
  H = Symbolics.hessian_sparsity(fun, ncon == 0 ? xsi : [xsi; ysi], full = false)
  H = ncon == 0 ? H : H[1:nvar,1:nvar]
  rows, cols, _ = findnz(H)
  vals = Symbolics.sparsehessian_vals(fun, xsi, rows, cols)
  nnzh = length(vals)
  # cfH is a Tuple{Expr, Expr}, cfH[2] is the in-place function
  # that we need to update a vector `vals` with the nonzeros of ∇²ℓ(x, y, μ).
  cfH = Symbolics.build_function(vals, xsi, ysi, μs, expression = Val{false})
  return SparseADHessian(nnzh, rows, cols, cfH[2])
end

function get_nln_nnzh(b::SparseADHessian, nvar)
  b.nnzh
end

function hess_structure!(
  b::SparseADHessian,
  nlp::ADModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  rows .= b.rows
  cols .= b.cols
  return rows, cols
end

function hess_coord!(
  b::SparseADHessian,
  nlp::ADModel,
  x::AbstractVector,
  y::AbstractVector,
  obj_weight::Real,
  vals::AbstractVector,
)
  @eval $(b.cfH)($vals, $x, $y, $obj_weight)
  return vals
end

function hess_coord!(
  b::SparseADHessian,
  nlp::ADModel,
  x::AbstractVector{T},
  obj_weight::Real,
  vals::AbstractVector,
) where T
  y = zeros(T, nlp.meta.ncon)
  @eval $(b.cfH)($vals, $x, $y, $obj_weight)
  return vals
end
