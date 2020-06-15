using Flux, ForwardDiff

struct FunctionPack
  f::Function
  ∇f::Function
  minimum::Array{Float32, 1}
end

function rosenbrock(x, a=1, b=100) 
  return (a -x[1])^2 + b*(x[2] - x[1]^2)^2
end

function rosenbrock_gradient(x, a=1, b=100)
  df_x = 4*b*x[1]^3 - 4*b*x[1]*x[2] + 2*x[1] - 2*a
  df_y = 2*b*(x[2] - x[1]^2)
  return [df_x, df_y]
end

rosenbrockPack(a=1, b=100) = (
  f = (x) -> rosenbrock(x, a, b),
  ∇f = (x) -> rosenbrock_gradient(x, a, b),
  minimum = (x) -> [1.0, 1.0]
)

function michalewicz(x, m=10)
  return -sum(sin(v)*sin(i*v^2/π)^(2m) for (i,v) in enumerate(x)) 
end

function michalewicz_gradient(x, m=10)
  ForwardDiff.gradient((v) -> michalewicz(v, m), x)
end

function michalewicz2_gradient(x, m=10)
  df_x = -(cos(x[1])*sin(x[1]^2/pi)^(2*m) + 4*m*x[1]*sin(x[1])*sin(x[1]^2/pi)^(-1 + 2*m)*cos(x[1]^2/pi)/pi)
  df_y = -(cos(x[2])*sin(2*x[2]^2/pi)^(2*m) + 8*m*x[2]*sin(x[2])*sin(2*x[2]^2/pi)^(-1 + 2*m)*cos(2*x[2]^2/pi)/pi)
  return [df_x, df_y]
end

michalewiczPack(m=10) = (
  (x) -> michalewicz(x, m),
  (x) -> michalewicz_gradient(x, m),
  (x) -> [2.20, 1.57]
)

michalewicz2Pack(m=10) = (
  (x) -> michalewicz(x, m),
  (x) -> michalewicz2_gradient(x, m),
  (x) -> [2.20, 1.57]
)

function wheeler(x, a=1.5) 
  return -exp(-(x[1]*x[2] - a)^2 -(x[2]-a)^2) 
end

function wheeler_gradient(x, a=1.5)
  df_x = 2*x[2]*(x[1]*x[2] - a)*-wheeler(x, a)
  df_y = (2*(x[1]^2 + 1)*x[2] - 2*a*(x[1]+1))*-wheeler(x, a)
  return [df_x, df_y]
end

wheelerPack(a) = (
  (x) -> wheeler(x, a),
  (x) -> wheeler_gradient(x, a),
  (x) -> [1.0, 1.5]
)

function sphere(x)
  sum(x.^2)
end

function sphere_gradient(x)
  x.*2
end

spherePack() = (
  sphere,
  sphere_gradient,
  (x) -> zeros(Float32, size(x)...)
)