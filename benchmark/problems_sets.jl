const meta = OptimizationProblems.meta
const nn = OptimizationProblems.default_nvar # 100 # default parameter for scalable problems

# Scalable problems from OptimizationProblem.jl
scalable_problems = meta[meta.variable_nvar .== true, :name] # problems that are scalable

all_problems = meta[meta.nvar .> 5, :name] # all problems with ≥ 5 variables
all_problems = setdiff(all_problems, scalable_problems) # avoid duplicate problems

# all scalable least squares problems with ≥ 5 variables
scalable_nls_problems = meta[(meta.variable_nvar .== true) .&& (meta.nvar .> 5) .&& (meta.objtype .== :least_squares), :name]

all_cons_problems = meta[(meta.nvar .> 5) .&& (meta.ncon .> 5), :name] # all problems with ≥ 5 variables
scalable_cons_problems = meta[(meta.variable_nvar .== true) .&& (meta.ncon .> 5), :name] # problems that are scalable
all_cons_problems = setdiff(all_cons_problems, scalable_cons_problems) # avoid duplicate problems

pre_problem_sets = Dict(
  "all" => all_problems, # all problems with ≥ 5 variables and not scalable
  "scalable" => scalable_problems, # problems that are scalable
  "all_cons" => all_cons_problems, # all problems with ≥ 5 variables anc cons and not scalable
  "scalable_cons" => scalable_cons_problems, # scalable problems with ≥ 5 variables anc cons
  "scalable_nls" => scalable_nls_problems,
)

# keys list all the accepted keywords to define backends
# values are generic backend to be used by default in this benchmark
all_backend_structure = Dict(
  "gradient_backend" => ADNLPModels.EmptyADbackend,
  "hprod_backend" => ADNLPModels.EmptyADbackend,
  "jprod_backend" => ADNLPModels.EmptyADbackend,
  "jtprod_backend" => ADNLPModels.EmptyADbackend,
  "jacobian_backend" => ADNLPModels.EmptyADbackend,
  "hessian_backend" => ADNLPModels.EmptyADbackend,
  "ghjvprod_backend" => ADNLPModels.EmptyADbackend,
  "hprod_residual_backend" => ADNLPModels.EmptyADbackend,
  "jprod_residual_backend" => ADNLPModels.EmptyADbackend,
  "jtprod_residual_backend" => ADNLPModels.EmptyADbackend,
  "jacobian_residual_backend" => ADNLPModels.EmptyADbackend,
  "hessian_residual_backend" => ADNLPModels.EmptyADbackend,
)

"""
    set_adnlp(pb::String, test_back::String, back_struct, n::Integer = nn, T::DataType = Float64)

Return an ADNLPModel with `back_struct` as an AD backend for `test_back ∈ keys(all_backend_structure)`
"""
function set_adnlp(
  pb::String,
  test_back::String, # backend specified
  back_struct,
  n::Integer = nn,
  T::DataType = Float64,
)
  pbs = Meta.parse(pb)
  backend_structure = Dict{String, Any}()
  for k in keys(all_backend_structure)
    if k == test_back
      push!(backend_structure, k => back_struct)
    else
      push!(backend_structure, k => all_backend_structure[k])
    end
  end
  return OptimizationProblems.ADNLPProblems.eval(pbs)(;
    type = T,
    n = n,
    gradient_backend = backend_structure["gradient_backend"],
    hprod_backend = backend_structure["hprod_backend"],
    jprod_backend = backend_structure["jprod_backend"],
    jtprod_backend = backend_structure["jtprod_backend"],
    jacobian_backend = backend_structure["jacobian_backend"],
    hessian_backend = backend_structure["hessian_backend"],
    ghjvprod_backend = backend_structure["ghjvprod_backend"],
  )
end

"""
    set_adnls(pb::String, test_back::String, back_struct, n::Integer = nn, T::DataType = Float64)

Return an ADNLSModel with `back_struct` as an AD backend for `test_back ∈ keys(all_backend_structure)`
"""
function set_adnls(
  pb::String,
  test_back::String, # backend specified
  back_struct,
  n::Integer = nn,
  T::DataType = Float64,
)
  pbs = Meta.parse(pb)
  backend_structure = Dict{String, Any}()
  for k in keys(all_backend_structure)
    if k == test_back
      push!(backend_structure, k => back_struct)
    else
      push!(backend_structure, k => all_backend_structure[k])
    end
  end
  return OptimizationProblems.ADNLPProblems.eval(pbs)(
    Val(:nls);
    type = T,
    n = n,
    gradient_backend = backend_structure["gradient_backend"],
    hprod_backend = backend_structure["hprod_backend"],
    jprod_backend = backend_structure["jprod_backend"],
    jtprod_backend = backend_structure["jtprod_backend"],
    jacobian_backend = backend_structure["jacobian_backend"],
    hessian_backend = backend_structure["hessian_backend"],
    ghjvprod_backend = backend_structure["ghjvprod_backend"],
    hprod_residual_backend = backend_structure["hprod_residual_backend"],
    jprod_residual_backend = backend_structure["jprod_residual_backend"],
    jtprod_residual_backend = backend_structure["jtprod_residual_backend"],
    jacobian_residual_backend = backend_structure["jacobian_residual_backend"],
    hessian_residual_backend = backend_structure["hessian_residual_backend"],
  )
end

function set_problem(
  pb::String,
  test_back::String,
  backend::String,
  s::String,
  n::Integer = nn,
  T::DataType = Float64,
)
  nlp = if backend == "jump"
    model = OptimizationProblems.PureJuMP.eval(Meta.parse(pb))(n = n)
    MathOptNLPModel(model)
  else
    set_adnlp(pb, f, test_back, backend, n, T)
  end
  return nlp
end
