using BenchmarkTools
using InteractiveUtils

include("lbfgs.jl")

include("functions.jl")
include("utils.jl")

opt = LBFGS(5)
f, ∇f, min = rosenbrockPack()
x = Float32[10.0, 10.0]
@code_warntype init!(opt, x)
init!(opt, x)
@code_warntype step!(opt, x, f, ∇f)
step!(opt, x, f, ∇f)