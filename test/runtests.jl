using Test

using ADNLPModels
using NLPModels

include("test-objectives.jl")

nvar = 10
model_fad = ADNLPModel(arglina, ones(nvar))
model_rad = RADNLPModel(arglina, ones(nvar))

@test model_rad.meta.nvar == nvar
@test model_rad.meta.nvar == model_fad.meta.nvar

x = ones(nvar); x[1:2:end] *= -1
@test obj(model_rad, x) ≈ 30
@test obj(model_rad, x) ≈ obj(model_fad, x)

@test grad(model_rad, x) ≈ grad(model_fad, x)

v = ones(nvar); v[2:2:end] *= -1
@test hprod(model_rad, x, v) ≈ hprod(model_fad, x, v)

