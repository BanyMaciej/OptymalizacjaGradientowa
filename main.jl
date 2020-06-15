using Pkg
Pkg.add("BenchmarkTools")
Pkg.add("ForwardDiff")
using BenchmarkTools
using InteractiveUtils

use_plots = true
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

function test(opt::DescentMethod, x::Array{Float32},  name)
  f, ∇f, min = rosenbrockPack()
  init!(opt, x)
  println(name * " optimize:")
  out = []
  @time n = optimize!(opt, x, f, ∇f, min(x))
  @show n
  @show x
end

function testDraw(opts::Array{DescentMethod})
  f, ∇f, m = michalewiczPack()
  plt = drawBackground(f)

  for opt in opts
    x = Float32[2.5, 2.0]
    init!(opt, x)
    result = optimizeWithSteps!(opt, x, f, ∇f, m(x), 1e-6)
    drawResult!(plt, result, typeof(opt))
  end
  savefig(plt, "plot.png")
  display(plt)
  gui()
  println("Press any key to continue")
  readline()
end

function testDrawLoss(opt::DescentMethod, x::Array{Float32})
  f, ∇f, m = rosenbrockPack()
  init!(opt, x)
  _minimum = m(x)
  result = optimizeWithSteps!(opt, x, f, ∇f, _minimum, 1e-4)
  loss = [mse(result[:, i], _minimum) for i in 1:size(result, 2)]
  plt = drawLoss(loss, typeof(opt))
  savefig(plt, "loss.png")
  display(plt)
  gui()
  println("Press any key to continue")
  readline()
end

test(Adam(), Float32[3.0, 2.0, 3.0, 2.0], "adam")
# test(Adam_old(), Float32[3.0, 2.0, 3.0, 2.0], "adam_old")
test(BFGS(), Float32[3.0, 2.0, 3.0, 2.0], "bfgs")
# test(BFGS_old(), Float32[3.0, 2.0, 3.0, 2.0], "bfgs_old")
test(LBFGS(5), Float32[3.0, 2.0, 3.0, 2.0], "lbfgs")
# test(LBFGS_old(5), Float32[3.0, 2.0, 3.0, 2.0], "lbfgs_old")


if use_plots
  testDrawLoss(Adam(), Float32[3.0, 2.0, 3.0, 2.0],)
  testDrawLoss(BFGS(), Float32[3.0, 2.0, 3.0, 2.0],)
  testDrawLoss(LBFGS(5), Float32[3.0, 2.0, 3.0, 2.0],)
  testDraw([Adam_old(), BFGS_old(), LBFGS_old(5)])
end