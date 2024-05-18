module ADNLPModelsEnzymeExt

using Enzyme, ADNLPModels

struct EnzymeADGradient <: ADNLPModels.ADBackend end

function EnzymeADGradient(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  x0::AbstractVector = rand(nvar),
  kwargs...,
)
  return EnzymeADGradient()
end

function ADNLPModels.gradient!(::EnzymeADGradient, g, f, x)
  Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Duplicated(x, g)) # gradient!(Reverse, g, f, x)
  return g
end

end