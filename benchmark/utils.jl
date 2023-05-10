function get_optimized_list(optimized_backend)
  return union(keys(optimized_backend), ["jump"])
end

is_jump_available(::Val{:jump}, T) = (T == Float64)
is_jump_available(::Val, T) = true

###################################################
#
#
#
###################################################
const meta = OptimizationProblems.meta
const nn = OptimizationProblems.default_nvar # 100 # default parameter for scalable problems

# Scalable problems from OptimizationProblem.jl
scalable_problems = meta[meta.variable_nvar .== true, :name] # problems that are scalable

all_problems = meta[meta.nvar .> 5, :name] # all problems with ≥ 5 variables
all_problems = setdiff(all_problems, scalable_problems) # avoid duplicate problems

all_cons_problems = meta[(meta.nvar .> 5) .&& (meta.ncon .> 5), :name] # all problems with ≥ 5 variables
scalable_cons_problems = meta[(meta.variable_nvar .== true) .&& (meta.ncon .> 5), :name] # problems that are scalable
all_cons_problems = setdiff(all_cons_problems, scalable_cons_problems) # avoid duplicate problems

pre_problem_sets = Dict(
  "all" => all_problems, # all problems with ≥ 5 variables and not scalable
  "scalable" => scalable_problems, # problems that are scalable
  "all_cons" => all_cons_problems, # all problems with ≥ 5 variables anc cons and not scalable
  "scalable_cons" => scalable_cons_problems, # scalable problems with ≥ 5 variables anc cons
)

###################################################
#
#
#
###################################################
benchmarked_optimized_backends = Dict(
  "gradient_backend" => Dict(
    "forward" => ADNLPModels.ForwardDiffADGradient,
    "reverse" => ADNLPModels.ReverseDiffADGradient,
    "enzyme" => EnzymeADGradient,
  ),
  "hprod_backend" => Dict( # 2 and 5 are best, maybe 5 more robust.
    "forward" => ADNLPModels.ForwardDiffADHvprod, # SPT 1
    "forwardSDT" => SPTADHvprod, # SPT 2 > STP 1
    #"forward1" => ForwardDiffADHvprod1,
    "forward2" => ForwardDiffADHvprod2,
    # "forward3" => ForwardDiffADHvprod3, # doesn't work
    #"forward4" => ForwardDiffADHvprod4,
    "forward5" => ForwardDiffADHvprod5,
  ),
  "jprod_backend" => Dict(
    "forward" => ADNLPModels.ForwardDiffADJprod, # use SparseDiffTools
    "reverse" => ADNLPModels.ReverseDiffADJprod,
    "forward1" => ForwardDiffADJprod1,
  ),
  "jtprod_backend" => Dict(
    "reverse" => OptimizedReverseDiffADJtprod,
    "forward" => OptimizedForwardDiffADJtprod,
  ),
  "jacobian_backend" => Dict(
    "sparse" => ADNLPModels.SparseADJacobian,
    # "forward" => ADNLPModels.ForwardDiffADJacobian, # slower
    # "reverse" => ADNLPModels.ReverseDiffADJacobian, # fails somehow
    # "zygote" => ADNLPModels.ZygoteADJacobian,
  ),
  "hessian_backend" => Dict(
    "sparse" => ADNLPModels.SparseADHessian,
    "forward" => ADNLPModels.ForwardDiffADHessian,
    # "sym" => ADNLPModels.SparseADHessian, # out of memory for large problems
  ),
  "ghjvprod_backend" => Dict(),
)

###################################################
#
#
#
###################################################
benchmarked_generic_backends = Dict(
  "gradient_backend" => Dict(
    "forward" => ADNLPModels.GenericForwardDiffADGradient,
    "reverse" => GenericReverseDiffADGradient,
    "zygote" => ADNLPModels.ZygoteADGradient,
  ),
  "hprod_backend" => Dict(
    "forward" => ADNLPModels.GenericForwardDiffADHvprod,
    "reverse" => ADNLPModels.ReverseDiffADHvprod,
  ),
  "jprod_backend" => Dict(
    "forward" => ADNLPModels.GenericForwardDiffADJprod,
    "reverse" => ADNLPModels.GenericReverseDiffADJprod,
    "zygote" => ADNLPModels.ZygoteADJprod,
  ),
  "jtprod_backend" => Dict(
    "forward" => ADNLPModels.ForwardDiffADJtprod,
    "reverse" => ADNLPModels.ReverseDiffADJtprod,
    "zygote" => ADNLPModels.ZygoteADJtprod,
  ),
  "jacobian_backend" => Dict(
    "forward" => ADNLPModels.ForwardDiffADJacobian,
    "reverse" => ADNLPModels.ReverseDiffADJacobian,
    "zygote" => ADNLPModels.ZygoteADJacobian,
    "sym" => ADNLPModels.SparseADJacobian, # out of memory for large problems
  ),
  "hessian_backend" => Dict(
    "forward" => ADNLPModels.ForwardDiffADHessian,
    "reverse" => ADNLPModels.ReverseDiffADHessian,
    "zygote" => ADNLPModels.ZygoteADHessian,
  ),
  "ghjvprod_backend" => Dict(
    "forward" => ADNLPModels.ForwardDiffADGHjvprod,
  ),
)

function set_back_list(::Val{:optimized}, test_back::String)
  return get_optimized_list(benchmarked_optimized_backends[test_back])
end

function get_back(::Val{:optimized}, test_back::String, backend::String)
  # test_back must be a key in benchmarked_optimized_backends
  # backend must be a key in benchmarked_optimized_backends[test_back]
  return benchmarked_optimized_backends[test_back][backend]
end

function set_back_list(::Val{:generic}, test_back::String)
  return keys(benchmarked_generic_backends[test_back])
end

function get_back(::Val{:generic}, test_back::String, backend::String)
  # test_back must be a key in benchmarked_generic_backends
  # backend must be a key in benchmarked_generic_backends[test_back]
  return benchmarked_generic_backends[test_back][backend]
end

# keys list all the accepted keywords to define backends
# values are generic backend to be used by default in this benchmark
all_backend_structure = Dict(
  "gradient_backend" => ADNLPModels.GenericForwardDiffADGradient,
  "hprod_backend" => ADNLPModels.ForwardDiffADHvprod,
  "jprod_backend" => ADNLPModels.ForwardDiffADJprod,
  "jtprod_backend" => ADNLPModels.ForwardDiffADJtprod,
  "jacobian_backend" => ADNLPModels.ForwardDiffADJacobian,
  "hessian_backend" => ADNLPModels.ForwardDiffADHessian,
  "ghjvprod_backend" => ADNLPModels.ForwardDiffADGHjvprod,
)

"""
Return an ADNLPModel with `back_struct` as an AD backend for `test_back ∈ keys(all_backend_structure)`
"""
function set_adnlp(pb::String, test_back::String, back_struct::Type{<:ADNLPModels.ADBackend}, n::Integer = nn, T::DataType = Float64)
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
    ;type = Val(T),
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

function set_adnlp(pb::String, f::String, test_back::String, backend::String, n::Integer = nn, T::DataType = Float64)
  back_struct = get_back(Val(Symbol(f)), test_back, backend)
  return set_adnlp(pb, test_back, back_struct, n, T)
end

function set_problem(pb::String, test_back::String, backend::String, f::String, s::String, n::Integer = nn, T::DataType = Float64)
  nlp = if backend == "jump"
    model = if s == "scalable"
      OptimizationProblems.PureJuMP.eval(Meta.parse(pb))(n = n)
    else
      OptimizationProblems.PureJuMP.eval(Meta.parse(pb))()
    end
    MathOptNLPModel(model)
  else
    set_adnlp(pb, f, test_back, backend, n, T)
  end
  return nlp
end
