struct DIADGradient{B, E} <: ADBackend
  backend::B
  prep::E
end

function DIADGradient(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  x0::AbstractVector = rand(nvar),
  backend = AutoReverseDiff(),
  kwargs...,
)
  prep = DifferentiationInterface.prepare_gradient(f, backend, x0)
  return DIADGradient(backend, prep)
end

function gradient(b::DIADGradient, f, x)
  g = DifferentiationInterface.gradient(f, b.prep, b.backend, x)
  return g
end

function gradient!(b::DIADGradient, g, f, x)
  DifferentiationInterface.gradient!(f, g, b.prep, b.backend, x)
  return g
end

struct DIADJprod{B, E} <: ADBackend
  backend::B
  prep::E
end

function DIADJprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  x0::AbstractVector = rand(nvar),
  backend = AutoReverseDiff(),
  kwargs...,
)
  dy = similar(x0, ncon)
  dx = similar(x0, nvar)
  prep = DifferentiationInterface.prepare_pushforward(c, dy, backend, x0, dx)
  return DIADJprod(backend, prep)
end

function Jprod!(b::DIADJprod, Jv, c, x, v, ::Val)
  DifferentiationInterface.pushforward!(c, Jv, b.prep, b.backend, x, v)
  return Jv
end

struct DIADJtprod{B, E} <: ADBackend
  backend::B
  prep::E
end

function DIADJtprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  x0::AbstractVector = rand(nvar),
  backend = AutoReverseDiff(),
  kwargs...,
)
  dx = similar(x0, nvar)
  dy = similar(x0, ncon)
  prep = DifferentiationInterface.prepare_pullback(c, dx, backend, x0, dy)
  return DIADJtprod(backend, prep)
end

function Jtprod!(b::DIADJtprod, Jtv, c, x, v, ::Val)
  DifferentiationInterface.pullback!(c, Jtv, b.prep, b.backend, x, v)
  return Jtv
end

struct DIADJacobian{B, E} <: ADBackend
  backend::B
  prep::E
end

function DIADJacobian(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  x0::AbstractVector = rand(nvar),
  backend = AutoForwardDiff(),
  kwargs...,
)
  y = similar(x0, ncon)
  prep = DifferentiationInterface.prepare_jacobian(c, y, backend, x0)
  return DIADJacobian(backend, prep)
end

function jacobian(b::DIADJacobian, c, x)
  J = DifferentiationInterface.jacobian(c, b.prep, b.backend, x)
  return J
end

struct SparseDIADJacobian{B, E} <: ADBackend
  backend::B
  prep::E
end

function SparseDIADJacobian(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  x0::AbstractVector = rand(nvar),
  coloring_algorithm::AbstractColoringAlgorithm = GreedyColoringAlgorithm{:direct}(
    postprocessing = true,
  ),
  detector::AbstractSparsityDetector = TracerSparsityDetector(),
  backend = AutoForwardDiff(),
  kwargs...,
)
  y = similar(x0, ncon)
  sparse_backend = DifferentiationInterface.AutoSparse(backend, sparsity_detector=detector, coloring_algorithm=coloring_algorithm)
  prep = DifferentiationInterface.prepare_jacobian(c, y, sparse_backend, x0)
  return SparseDIADJacobian(sparse_backend, prep)
end

function jacobian(b::SparseDIADJacobian, c, x)
  J = DifferentiationInterface.jacobian(c, b.prep, b.backend, x)
  return J
end

struct DIADHvprod{B, E} <: ADBackend
  backend::B
  prep::E
end

function DIADHvprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  x0::AbstractVector = rand(nvar),
  backend = AutoReverseDiff(),
  kwargs...,
)
  tx = similar(x0)
  prep = DifferentiationInterface.prepare_hvp(f, backend, x0, tx)
  return DIADHvprod(backend, prep)
end

function Hvprod!(b::DIADHvprod, Hv, f, x, v, ::Val)
  DifferentiationInterface.hvp!(f, Hv, b.prep, b.backend, x, v)
  return Hv
end

struct DIADHessian{B, E} <: ADBackend
  backend::B
  prep::E
end

function DIADHessian(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  x0::AbstractVector = rand(nvar),
  first_backend = AutoReverseDiff(),
  second_backend = AutoForwardDiff(),
  kwargs...,
)
  backend = DifferentiationInterface.SecondOrder(second_backend, first_backend)
  prep = DifferentiationInterface.prepare_hessian(f, backend, x0)
  return DIADHessian(backend, prep)
end

function hessian(b::DIADHessian, f, x)
  H = DifferentiationInterface.hessian(f, b.prep, b.backend, x)
  return H
end

struct SparseDIADHessian{B, E} <: ADBackend
  backend::B
  prep::E
end

function SparseDIADHessian(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  x0::AbstractVector = rand(nvar),
  coloring_algorithm::AbstractColoringAlgorithm = GreedyColoringAlgorithm{:substitution}(
    postprocessing = true,
  ),
  detector::AbstractSparsityDetector = TracerSparsityDetector(),
  first_backend = AutoReverseDiff(),
  second_backend = AutoForwardDiff(),
  kwargs...,
)
  backend = DifferentiationInterface.SecondOrder(second_backend, first_backend)
  sparse_backend = DifferentiationInterface.AutoSparse(backend, sparsity_detector=detector, coloring_algorithm=coloring_algorithm)
  prep = DifferentiationInterface.prepare_hessian(f, backend, x0)
  return SparseDIADHessian(sparse_backend, prep)
end

function hessian(b::SparseDIADHessian, f, x)
  H = DifferentiationInterface.hessian(f, b.prep, b.backend, x)
  return H
end
