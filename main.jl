#using Pkg
#Pkg.add("BenchmarkTools")
#using BenchmarkTools


use_plots = true
if use_plots
  include("draw.jl")
end

include("Adam.jl")
# include("Adam copy.jl")
include("Adam_old.jl")
include("bfgs.jl")
include("lbfgs.jl")

# funkcja rosenbrock'a, x - wektor wejściowy dwuwymiarowy, jakby (x, y)
function rosenbrock(x, a=1, b=100) 
  return (a -x[1])^2 + b*(x[2] - x[1]^2)^2
end

function rosenbrock_gradient(x, a=1, b=100)
  df_x = 4*b*x[1]^3 - 4*b*x[1]*x[2] + 2*x[1] - 2*a
  df_y = 2*b*(x[2] - x[1]^2)
  return [df_x, df_y]
end

function michalewicz(x, m=10)
  return -sum(sin(v)*sin(i*v^2/π)^(2m)   for 
    (i,v) in enumerate(x)) 
end

function michalewicz2_gradient(x, m=10)
  df_x = -(cos(x[1])*sin(x[1]^2/pi)^(2*m) + 4*m*x[1]*sin(x[1])*sin(x[1]^2/pi)^(-1 + 2*m)*cos(x[1]^2/pi)/pi)
  df_y = -(cos(x[2])*sin(2*x[2]^2/pi)^(2*m) + 8*m*x[2]*sin(x[2])*sin(2*x[2]^2/pi)^(-1 + 2*m)*cos(2*x[2]^2/pi)/pi)
  return [df_x, df_y]
end

function wheeler(x, a=1.5) 
  return -exp(-(x[1]*x[2] - a)^2 -(x[2]-a)^2) 
end

function wheeler_gradient(x, a=1.5)
  df_x = 2*x[2]*(x[1]*x[2] - a)*-wheeler(x, a)
  df_y = (2*(x[1]^2 + 1)*x[2] - 2*a*(x[1]+1))*-wheeler(x, a)
  return [df_x, df_y]
end

function adamOptimize(x, f, ∇f, return_steps=false)
  M = Adam()
  init!(M, x)
  out = similar(x)
  copyto!(out, x)
  if return_steps
    temp = out
    for i = 1:10^4
      step!(M, out, f, ∇f)
      println(out)
      temp = [temp out]
    end
    return temp
  else
    for i = 1:10^6
      step!(M, out, f, ∇f)
    end
    return out

  end
end

function adamOptimizeOld(x, f, ∇f, return_steps=false)
  M = Adam_old()
  init!(M, x)
  out = similar(x)
  copyto!(out, x)
  if return_steps
    temp = out
    for i = 1:10^4
      out .= step!(M, out, f, ∇f)
      temp = [temp out]
    end
    return temp
  else
    for i = 1:10^6
      out .= step!(M, out, f, ∇f)
    end
    return out
  end
end

function bfgsOptimize(x, f, ∇f, return_steps=false)
  M = BFGS(x) 
  if return_steps
    temp = x
    for i = 1:1000
      step!(M, x, f, ∇f)
      temp = [temp x]
    end
    return temp
  else
    for i = 1:1000
      step!(M, x, f, ∇f)
    end
    return x
  end
end

function lbfgsOptimize(f, ∇f, x, m, return_steps=false)
  M = LBFGS()
  lbfgs_init!(M, m)
  if return_steps
    temp = x
    for i = 1:50000
      temp = [temp lbfgs_step!(M , f, ∇f, x)]
    end
    return temp
  else
    for i = 1:50000
      x = lbfgs_step!(M , f, ∇f, x)
    end
    return x
  end
end

f, ∇f = rosenbrock, rosenbrock_gradient


x = [3.0, 2.0]
#println("Adam new:")
#@time adamOptimize(x, f, ∇f) 
#println("Adam old:")
#@time adamOptimizeOld(x, f, ∇f)


# println("Adam old:")
# @btime adamOptimizeOld($x, $f, $∇f)

# println("BFGS new:")
# @time out = bfgsOptimize(x, f, ∇f)
# println(out)

# println("BFGS:")
# @time out = bfgsOptimize(f, ∇f, x)
# println(out)
# println("L-BFGS, m=10: ")
# @time out = lbfgsOptimize(f, ∇f, x, 10)
# println(out)
# println("L-BFGS, m=5: ")
# @time out = lbfgsOptimize(f, ∇f, x, 5)
# println(out)
# println("L-BFGS, m=2: ")
# @time out = lbfgsOptimize(f, ∇f, x, 2)
# println(out)
# println("L-BFGS, m=1: ")
# @time out = lbfgsOptimize(f, ∇f, x, 1)
# println(out)
#
println("start")
if use_plots
  plt = drawBackground(f)
  resultAdam = adamOptimize(x, f, ∇f, true)
  drawResult!(plt, resultAdam)
  savefig(plt, "plot.png")
end
