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

@init begin
  @require Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9" begin
    function ADNLPModels.gradient!(::EnzymeADGradient, g, f, x)
      Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Duplicated(x, g)) # gradient!(Reverse, g, f, x)
      return g
    end
  end
end
