# import Pkg
# Pkg.add("Plots")
# Pkg.add("PyPlot")
using Plots
pyplot()

function drawResult!(plt, result, name)
  plot!(plt, result[1, :], result[2, :], label=name)
end

function drawLoss(loss, name)
  x = 1:length(loss)
  plot(x, loss, label=name, show=true)
end

function drawBackground(f)
  x = -3.0:0.05:3.0;
  y = -3.0:0.05:3.0;
  contour(x, y, (a,b)->f([a, b]), fill=true, show=true)
end