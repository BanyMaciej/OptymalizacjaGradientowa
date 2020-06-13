# using Pkg
# Pkg.add("Flux")
# Pkg.add("Plots")
# Pkg.add("PyPlot")
include("Adam.jl")
using Plots
pyplot()
using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle, Params
using Base.Iterators: repeated
using Flux: @epochs


function train!(m, loss, data, opt::DescentMethod, cb = () -> ())
  ps = Params(params(m))
  for d in data
    if d isa AbstractArray{<:Number}
      gs = gradient(ps) do
        loss(d)
      end
    else
      gs = gradient(ps) do
        loss(d...)
      end
    end
    for x in ps
      gs[x] == nothing && continue
      init!(opt, x)
      step!(opt, x, (x) -> nothing, (x)->gs[x])
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
  
  losses = []
  push!(losses, losser(X, Y))
  @show typeof(losses)
  callback = () -> (l = losser(X, Y); push!(losses, l); @show(l))

  @epochs epochs train!(m, losser, dataset, optimizer, callback);

  @show accuracy(m, X, Y)
  return (m, losses)
end

m, loss = test_training(mse, Adam(), 200)

scatter(loss[1, :])

test_X = hcat(float.(reshape.(Flux.Data.MNIST.images(:test), :))...)
test_Y = onehotbatch(Flux.Data.MNIST.labels(:test), 0:9);
@show accuracy(m, test_X, test_Y)

readline()