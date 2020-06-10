# funkcja rosenbrock'a, x - wektor wejściowy dwuwymiarowy, jakby (x, y)
function rosenbrock(x, a=1, b=100) 
  return (a -x[1])^2 + b*(x[2] - x[1]^2)^2
end

function rosenbrock_gradient(x, a=1, b=100)
  df_x = 4*b*x[1]^3 - 4*b*x[1]*x[2] + 2*x[1] - 2*a
  df_y = 2*b*(x[2] - x[1]^2)
  return [df_x, df_y]
end


function rosenbrock_minimum()
  return [1.0, 1.0]
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