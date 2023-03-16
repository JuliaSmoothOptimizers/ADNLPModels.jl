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
