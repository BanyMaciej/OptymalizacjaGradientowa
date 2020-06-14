using Pkg
Pkg.add("BenchmarkTools")
using BenchmarkTools
using InteractiveUtils

use_plots = false
if use_plots
  include("draw.jl")
end

include("Adam.jl")
include("Adam_old.jl")
include("bfgs.jl")
include("bfgs_old.jl")
include("lbfgs.jl")
include("lbfgs_old.jl")

include("functions.jl")
include("utils.jl")

function shouldStop(act, dest, ϵ)
  mse(act, dest) <= ϵ
end

function optimize!(M::DescentMethod, x, f, ∇f, minimum, ϵ = 1e-6)
  n = 0
  while n < 500000
    n += 1
    step!(M, x, f, ∇f)
    shouldStop(x, minimum, ϵ) && break
  end
  return n
end

function optimizeWithSteps!(M::DescentMethod, x, f, ∇f, minimum, ϵ = 1e-6)
  temp = deepcopy(x)
  n = 0
  while n < 50000
    n += 1
    step!(M, x, f, ∇f)
    temp = [temp x]
    shouldStop(x, minimum, ϵ) && break
  end
  return temp
end

function test(opt::DescentMethod, name)
  f, ∇f, min = rosenbrockPack()
  x = Float32[10.0, 10.0]
  init!(opt, x)
  println(name * " optimize:")
  out = []
  @time n = optimize!(opt, x, f, ∇f, min)
  @show n
  @show x
end

function testDraw(opts::Array{DescentMethod})
  f, ∇f, min = rosenbrockPack()
  plt = drawBackground(f)

  for opt in opts
    x = Float32[2.0, 2.0]
    init!(opt, x)
    result = optimizeWithSteps!(opt, x, f, ∇f, min, 1e-4)
    drawResult!(plt, result, typeof(opt))
  end
  savefig(plt, "plot.png")
  display(plt)
  gui()
  readline()
end

test(Adam(), "adam")
test(Adam_old(), "adam_old")
test(BFGS(), "bfgs")
test(BFGS_old(), "bfgs_old")
test(LBFGS(5), "lbfgs")
test(LBFGS_old(5), "lbfgs_old")


if use_plots
  testDraw([Adam(), BFGS(), LBFGS(5)])
end