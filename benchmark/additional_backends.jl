struct GenericForwardDiffADGradient <: ADNLPModels.ADBackend end
function GenericForwardDiffADGradient(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  x0::AbstractVector = rand(nvar),
  kwargs...,
)
  return GenericForwardDiffADGradient()
end
ADNLPModels.gradient(::GenericForwardDiffADGradient, f, x) = ForwardDiff.gradient(f, x)
function ADNLPModels.gradient!(::GenericForwardDiffADGradient, g, f, x)
  return ForwardDiff.gradient!(g, f, x)
end

struct GenericReverseDiffADGradient <: ADNLPModels.ADBackend end
function GenericReverseDiffADGradient(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  x0::AbstractVector = rand(nvar),
  kwargs...,
)
  return GenericReverseDiffADGradient()
end
ADNLPModels.gradient(::GenericReverseDiffADGradient, f, x) = ReverseDiff.gradient(f, x)
function ADNLPModels.gradient!(::GenericReverseDiffADGradient, g, f, x)
  return ReverseDiff.gradient!(g, f, x)
end
