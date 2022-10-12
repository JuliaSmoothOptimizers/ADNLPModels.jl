struct SparseADJacobian <: ADBackend
  nnzj
  rows
  cols
  cfJ
end

function SparseADJacobian(nvar, f, ncon, c; kwargs...)
  @variables xs[1:nvar]
  w = Symbolics.scalarize(xs)
  _fun = c(w)
  S = Symbolics.jacobian_sparsity(_fun, w)
  rows, cols, _ = findnz(S)
  vals = Symbolics.sparsejacobian_vals(_fun, w, rows, cols)
  nnzj = length(rows)
  cfJ = Symbolics.build_function(vals, w, expression = Val{false})
  SparseADJacobian(nnzj, rows, cols, cfJ)
end

function get_nln_nnzj(b::SparseADJacobian, nvar, ncon)
  b.nnzj
end

function jac_structure!(
  b::SparseADJacobian,
  nlp,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  rows .= b.rows
  cols .= b.cols
  return rows, cols
end
function jac_coord!(b::SparseADJacobian, nlp, x::AbstractVector, vals::AbstractVector)
  _fun = eval(b.cfJ[2])
  Base.invokelatest(_fun, vals, x)
  return vals
end
