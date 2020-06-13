# import Pkg
# Pkg.add("Plots")
# Pkg.add("PyPlot")
using Plots
pyplot()

function drawResult!(plt, result, kwargs...)
  plot!(plt, result[1, :], result[2, :], kwargs...)
end

function drawBackground(f)
  x = 0.5:0.1:3;
  y = 0.5:0.1:4;
  contour(x, y, (a,b)->f([a, b]), fill=true, show=true)
end