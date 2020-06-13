using LinearAlgebra 
import Base.MathConstants: φ

abstract type DescentMethod end

function line_search(f, f′, d)
  a, b = bracket_minimum(f)
  x, y = golden_section_search(f, a, b)
  return x/2 + y/2
end

function bracket_minimum(f, x=0; s=1e-2, k=2.0, ϵ=1e-6)
  a, ya = x, f(x)
  b, yb = a + s, f(a + s)
  if yb > ya
    a, b = b, a
    ya, yb = yb, ya
    s = -s
  end
  i = 0
  while true
    c, yc = b + s, f(b + s)
    if yc - yb > ϵ
      return a < c ? (a, c) : (c, a)
    end
    a, ya, b, yb = b, yb, c, yc
    s *= k
    i += 1
  end
end

function golden_section_search(f, a, b, n=50)
  ρ = φ-1
  d = ρ * b + (1 - ρ)*a
  yd = f(d)
  for i = 1 : n-1
    c = ρ*a + (1 - ρ)*b
    yc = f(c)
    if yc < yd
      b, d, yd = d, c, yc
      else
      a, b = b, c
    end
  end
  return a < b ? (a, b) : (b, a)
end

function bisection(f′, a, b, ϵ=1)
  if a > b; a,b = b,a; end # ensure a < b
  ya, yb = f′(a), f′(b)
  if ya == 0; b = a; end
  if yb == 0; a = b; end
  while b - a > ϵ
    x = (a+b)/2
    y = f′(x)
    if y == 0
      a, b = x, x
    elseif sign(y) == sign(ya)
      a = x
    else
      b = x
    end
  end
  return (a,b)
end

function mse(y_pred, y) 
  mean((y - y_pred) .^ 2)
end