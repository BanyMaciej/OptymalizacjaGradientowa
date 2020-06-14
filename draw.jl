import Pkg
Pkg.add("Plots")
Pkg.add("PyPlot")
using Plots
pyplot()

function drawResult!(plt, result, name)
  plot!(plt, result[1, :], result[2, :], label=name)
end

function drawBackground(f)
  x = -2.0:0.05:2.0;
  y = -0.5:0.05:3.5;
  contour(x, y, (a,b)->f([a, b]), fill=true, show=true)
end