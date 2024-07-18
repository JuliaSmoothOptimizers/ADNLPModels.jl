using Documenter, ADNLPModels

makedocs(
  modules = [ADNLPModels],
  doctest = true,
  linkcheck = false,
  format = Documenter.HTML(assets = ["assets/style.css"],
                           ansicolor = true,
                           prettyurls = get(ENV, "CI", nothing) == "true"),
  sitename = "ADNLPModels.jl",
  pages = [
    "Home" => "index.md",
    "Tutorial" => "tutorial.md",
    "Backend" => "backend.md",
    "Default backends" => "predefined.md",
    "Build a hybrid NLPModel" => "mixed.md",
    "Support multiple precision" => "generic.md",
    "Sparse Jacobian and Hessian" => "sparse.md",
    "Performance tips" => "performance.md",
    "Reference" => "reference.md",
  ],
)

deploydocs(
  repo = "github.com/JuliaSmoothOptimizers/ADNLPModels.jl.git",
  push_preview = true,
  devbranch = "main",
)
