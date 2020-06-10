# using Pkg
# Pkg.add("BenchmarkTools")
using BenchmarkTools


use_plots = true
if use_plots
  include("draw.jl")
end

include("Adam.jl")
# include("Adam copy.jl")
include("Adam_old.jl")
include("bfgs.jl")
include("lbfgs.jl")

include("functions.jl")
include("utils.jl")

function shouldStop(act, dest, ϵ)
  norm(dest-act) <= ϵ
end

ϵ = 1e-8

function optimize(M::DescentMethod, x, f, ∇f, minimum, ϵ = 1e-6; return_steps=false)
  init!(M, x)
  out = similar(x)
  copyto!(out, x)
  if return_steps
    temp = out
    n = 0
    while n < 50000
      n += 1
      step!(M, out, f, ∇f)
      temp = [temp out]
      shouldStop(out, minimum, ϵ) && break
    end
    return (temp, n)
  else
    n = 0
    while n < 500000
      n += 1
      step!(M, out, f, ∇f)
      shouldStop(out, minimum, ϵ) && break
    end
    return (out, n)
  end
end

f, ∇f, min = rosenbrock, rosenbrock_gradient, rosenbrock_minimum()

x = [10.0, 10.0]

# println("LBFGS-10 optimize:")
# @btime optimize(LBFGS(10), x, f, ∇f, min)
# println("LBFGS-5 optimize:")
# @btime optimize(LBFGS(5), x, f, ∇f, min)
# println("LBFGS-2 optimize:")
# @btime optimize(LBFGS(2), x, f, ∇f, min)
# println("LBFGS-1 optimize:")
# @btime optimize(LBFGS(1), x, f, ∇f, min)
# println("BFGS optimize:")
# @btime optimize(BFGS(), x, f, ∇f, min)
# println("Adam optimize:")
# @btime optimize(Adam(), x, f, ∇f, min)

if use_plots
  plt = drawBackground(f)
  resultAdam, nAdam = optimize(Adam(), x, f, ∇f, min; return_steps=true)
  resultBfgs, nBfgs = optimize(BFGS(), x, f, ∇f, min; return_steps=true)
  resultLBfgs, nLBfgs = optimize(LBFGS(5), x, f, ∇f, min; return_steps=true)
  drawResult!(plt, resultAdam)
  savefig(plt, "plot.png")
end
