struct SparseADJacobian{J} <: ADBackend
  nnzj::Int
  rows::Vector{Int}
  cols::Vector{Int}
  cfJ::J
end

function SparseADJacobian(nvar, f, ncon, c!; kwargs...)
  @variables xs[1:nvar] out[1:ncon]
  wi = Symbolics.scalarize(xs)
  wo = Symbolics.scalarize(out)
  _fun = c!(wo, wi)
  S = Symbolics.jacobian_sparsity(c!, wo, wi)
  rows, cols, _ = findnz(S)
  vals = Symbolics.sparsejacobian_vals(_fun, wi, rows, cols)
  nnzj = length(rows)
  cfJ = Symbolics.build_function(vals, wi, expression = Val{false})
  SparseADJacobian(nnzj, rows, cols, cfJ)
end

function get_nln_nnzj(b::SparseADJacobian, nvar, ncon)
  b.nnzj
end

function jac_structure!(
  b::SparseADJacobian,
  ::ADModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  rows .= b.rows
  cols .= b.cols
  return rows, cols
end
function jac_coord!(b::SparseADJacobian, ::ADModel, x::AbstractVector, vals::AbstractVector)
  _fun = eval(b.cfJ[2])
  Base.invokelatest(_fun, vals, x)
  return vals
end
