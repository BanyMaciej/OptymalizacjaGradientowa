# using Pkg
# Pkg.add("Flux")
# Pkg.add("Plots")
# Pkg.add("PyPlot")
include("Adam.jl")
include("BFGS.jl")
# using Plots
# pyplot()
using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle, Params
using Base.Iterators: repeated
using Flux: @epochs

using InteractiveUtils

function train!(m, loss, data, opt::DescentMethod, cb = () -> ())
  ps = params(m)
  for d in data
    gs = gradient(ps) do
      loss(d...)
    end
    for x in ps
      weightSize = size(x)
      gs[x] == nothing && continue
      f(o) = (temp = copy(x); x = o; out = loss(d...); x = temp; return out)
      ∇f(x) = (println("gradient"); return gs[x])
      init!(opt, x)
      step!(opt, x, f, ∇f)
    end
    cb()
  end
end

function accuracy(m, x, y)
  mean(onecold(m(x)) .== onecold(y)) 
end

function test_training(loss, optimizer::DescentMethod, epochs::Int = 20)  
  imgs = Flux.Data.MNIST.images()
  labels = Flux.Data.MNIST.labels();
  X = hcat(float.(reshape.(imgs, :))...) 
  Y = onehotbatch(labels, 0:9)
  m = Chain(
    Dense(28^2, 32, relu),
    Dense(32, 10),
    softmax)
  dataset = repeated((X,Y),1)

  losser = (x, y) -> loss(m(x), y)
  
  losses = Float32[]
  push!(losses, losser(X, Y))
  callback = () -> (l = losser(X, Y); push!(losses, l); @show(l))

  @epochs epochs train!(m, losser, dataset, optimizer, callback);

  @show accuracy(m, X, Y)
  return (m, losses)
end

 test(opt::DescentMethod) = (@time test_training(mse, opt, 20))

 m, loss = test(BFGS())

# scatter(loss[1, :])

# test_X = hcat(float.(reshape.(Flux.Data.MNIST.images(:test), :))...)
# test_Y = onehotbatch(Flux.Data.MNIST.labels(:test), 0:9);
# @show accuracy(m, test_X, test_Y)

# readline()