using Documenter, ADNLPModels

makedocs(
  modules = [ADNLPModels],
  doctest = true,
  linkcheck = false,
  strict = true,
  format = Documenter.HTML(
    assets = ["assets/style.css"],
    prettyurls = get(ENV, "CI", nothing) == "true",
  ),
  sitename = "ADNLPModels.jl",
  pages = [
    "Home" => "index.md",
    "Tutorial" => "tutorial.md",
    "Backend" => "backend.md",
    "Default backends" => "predefined.md",
    "Support multiple precision" => "generic.md",
    "Reference" => "reference.md",
  ],
)

deploydocs(
  repo = "github.com/JuliaSmoothOptimizers/ADNLPModels.jl.git",
  push_preview = true,
  devbranch = "main",
)
