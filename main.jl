# using Pkg
# Pkg.add("BenchmarkTools")
using BenchmarkTools


use_plots = true
if use_plots
  include("draw.jl")
end

include("Adam.jl")
include("Adam_old.jl")
include("bfgs.jl")
include("bfgs_old.jl")
include("lbfgs.jl")

include("functions.jl")
include("utils.jl")

function shouldStop(act, dest, ϵ)
  norm(dest-act) <= ϵ
end

ϵ = 1e-8

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
  temp = [deepcopy(x)]
  n = 0
  while n < 50000
    n += 1
    step!(M, x, f, ∇f)
    push!(temp, x)
    shouldStop(x, minimum, ϵ) && break
  end
  return temp
end

function test(opt::DescentMethod, name)
  f, ∇f, min = rosenbrock, rosenbrock_gradient, rosenbrock_minimum()
  x = [10.0, 10.0]
  init!(opt, x)
  println(name * " optimize:")
  @time n = optimize!(opt, x, f, ∇f, min)
  @show n
  @show x
end

function testDraw(opt::DescentMethod, name)
  f, ∇f, min = rosenbrock, rosenbrock_gradient, rosenbrock_minimum()
  x = [10.0, 10.0]
  plt = drawBackground(f)
  init!(opt, x)
  result = optimizeWithSteps!(opt, x, f, ∇f, min)
  drawResult!(plt, result)
  savefig(plt, "plot.png")
  display(plt)
  gui()
  readline()
end

test(Adam(), "Adam")

if use_plots
  testDraw(Adam(), "Adam")
end