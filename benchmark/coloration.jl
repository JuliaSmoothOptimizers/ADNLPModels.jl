using Pkg
Pkg.activate("coloration")
Pkg.add(url="https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl", rev="main")
using ADNLPModels, OptimizationProblems, NLPModels
using Symbolics # only option to compute sparsity pattern
# coloration packages
using SparseDiffTools, ColPack
using BenchmarkTools

# Problem set
const meta = OptimizationProblems.meta
const nn = OptimizationProblems.default_nvar # 100 # default parameter for scalable problems

# Scalable problems from OptimizationProblem.jl
scalable_problems = meta[meta.variable_nvar .== true, :name] # problems that are scalable

all_problems = meta[meta.nvar .> 5, :name] # all problems with ≥ 5 variables
all_problems = setdiff(all_problems, scalable_problems) # avoid duplicate problems

all_cons_problems = meta[(meta.nvar .> 5) .&& (meta.ncon .> 5), :name] # all problems with ≥ 5 variables
scalable_cons_problems = meta[(meta.variable_nvar .== true) .&& (meta.ncon .> 5), :name] # problems that are scalable
all_cons_problems = setdiff(all_cons_problems, scalable_cons_problems) # avoid duplicate problems

result = Dict()
for pb in scalable_cons_problems
    nscal = 100
    n = eval(Meta.parse("OptimizationProblems.get_" * pb * "_nvar(n = $(nscal))"))
    ncon = eval(Meta.parse("OptimizationProblems.get_" * pb * "_nnln(n = $(nscal))"))
    T = Float64
    backend_structure = Dict(
    "gradient_backend" => ADNLPModels.EmptyADbackend, # ADNLPModels.GenericForwardDiffADGradient,
    "hprod_backend" => ADNLPModels.EmptyADbackend, # ADNLPModels.ForwardDiffADHvprod,
    "jprod_backend" => ADNLPModels.EmptyADbackend, # ADNLPModels.ForwardDiffADJprod,
    "jtprod_backend" => ADNLPModels.EmptyADbackend, # ADNLPModels.ForwardDiffADJtprod,
    "jacobian_backend" => ADNLPModels.EmptyADbackend, # ADNLPModels.ForwardDiffADJacobian,
    "hessian_backend" => ADNLPModels.EmptyADbackend, # ADNLPModels.ForwardDiffADHessian,
    "ghjvprod_backend" => ADNLPModels.EmptyADbackend, # ADNLPModels.ForwardDiffADGHjvprod,
    )
    nlp = OptimizationProblems.ADNLPProblems.eval(Meta.parse(pb))(
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
    @info "Coloration test $(pb) of size n=$n / $(nlp.meta.nvar) and ncon=$ncon / $(nlp.meta.nnln)"
    # test jacobian coloration
    c! = nlp.c!
    x0 = nlp.meta.x0
    output = similar(x0, nlp.meta.nnln)
    J = Symbolics.jacobian_sparsity(c!, output, x0)
    if J != spzeros(size(J))
        list = Dict()
        # ::SparseDiffTools.SparseDiffToolsColoringAlgorithm
        function SDTcoloring(J)
            alg = SparseDiffTools.GreedyD1Color()
            colors = matrix_colors(J, alg)
            ncolors = maximum(colors)
            return ncolors
        end
        ncolorsSDT = SDTcoloring(J)
        list["SDT"] = ncolorsSDT

        # ColPack
        function ColPackcoloring(J, coloring, ordering, partition_by_rows)
            adjA = ColPack.matrix2adjmatrix(J) # ; partition_by_rows=false
            CPC = ColPackColoring(adjA, coloring, ordering)
            colors = get_colors(CPC)
            ncolors = length(unique(colors))
            return ncolors
        end
        for ordering in ColPack.ORDERINGS, coloring in ColPack.COLORINGS, partition_by_rows in (true, false)
            ncolors = ColPackcoloring(J, coloring, ordering, partition_by_rows)
            list["($ordering, $coloring, $(partition_by_rows))"] = ncolors
            # @info "ColPack with (ordering=$ordering, coloring=$coloring, rows=$(partition_by_rows)) gave $ncolors colors"
        end
        @show findmin(list)
        result[pb] = list
    else
        @info "0 elements in the matrix"
    end
end

# test Hessian coloration
# TODO

#=
[ Info: Coloration test camshape of size n=100 / 100 and ncon=203 / 203
findmin(list) = (3, "SDT")
ncolorsSDT = 3
[ Info: Coloration test clnlbeam of size n=99 / 99 and ncon=32 / 32
findmin(list) = (2, "(distance_two_smallest_last_ordering(\"DISTANCE_TWO_SMALLEST_LAST\"), acyclic_coloring(\"ACYCLIC\"), false)")
ncolorsSDT = 4
[ Info: Coloration test controlinvestment of size n=100 / 100 and ncon=49 / 49
findmin(list) = (2, "(distance_two_smallest_last_ordering(\"DISTANCE_TWO_SMALLEST_LAST\"), acyclic_coloring(\"ACYCLIC\"), false)")
ncolorsSDT = 4
[ Info: Coloration test elec of size n=99 / 99 and ncon=33 / 33
findmin(list) = (1, "(incidence_degree_ordering(\"INCIDENCE_DEGREE\"), star_coloring(\"STAR\"), false)")
ncolorsSDT = 3
[ Info: Coloration test hovercraft1d of size n=98 / 95 and ncon=0 / 0
[ Info: 0 elements in the matrix
[ Info: Coloration test polygon of size n=100 / 100 and ncon=1225 / 1225
findmin(list) = (53, "(dynamic_largest_first_ordering(\"DYNAMIC_LARGEST_FIRST\"), d1_coloring(\"DISTANCE_ONE\"), true)")
ncolorsSDT = 100
[ Info: Coloration test polygon1 of size n=100 / 100 and ncon=0 / 0
[ Info: 0 elements in the matrix
[ Info: Coloration test polygon3 of size n=100 / 100 and ncon=100 / 100
findmin(list) = (3, "(distance_two_smallest_last_ordering(\"DISTANCE_TWO_SMALLEST_LAST\"), acyclic_coloring(\"ACYCLIC\"), false)")
ncolorsSDT = 4
[ Info: Coloration test robotarm of size n=109 / 118 and ncon=102 / 111
findmin(list) = (6, "SDT")
ncolorsSDT = 6
[ Info: Coloration test structural of size n=600 / 2256 and ncon=0 / 0
[ Info: 0 elements in the matrix

=#