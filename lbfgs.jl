using LinearAlgebra
include("utils.jl")

mutable struct LBFGS <: DescentMethod
  m 
  δs 
  γs 
  qs 
  LBFGS(m) = (x = new(); x.m = m; return x)
end 
function init!(M::LBFGS, θ; m = 1) 
  M.m = m 
  M.δs = [] 
  M.γs = [] 
  M.qs = [] 
  return M 
end 
function step!(M::LBFGS, θ, f, ∇f) 
  δs, γs, qs = M.δs, M.γs, M.qs
  m, g = length(δs),  ∇f(θ)
  d = -g
  if m > 0 
    q = g 
    for i in m : -1 : 1 
      qs[i] = copy(q) 
      q -= (δs[i]⋅q)/(γs[i]⋅δs[i])*γs[i] 
    end 
    z = (γs[m] .* δs[m] .* q) / (γs[m]⋅γs[m]) 
    for i in 1 : +1 : m 
      z += δs[i]*(δs[i]⋅qs[i] - γs[i]⋅z)/(γs[i]⋅δs[i]) 
    end 
    d = -z;
  end 
  ϕ  = α ->  f(θ + α*d);  ϕ′ = α -> ∇f(θ + α*d)⋅d  
  α  = line_search(ϕ, ϕ′, d)  
  θ′ = θ + α*d; g′ = ∇f(θ′)  
  δ  = θ′ - θ; γ  = g′ - g  
  push!(δs, δ);    push!(γs, γ);  push!(qs, zero(θ))  
  while length(δs) > M.m    
    popfirst!(δs); popfirst!(γs); popfirst!(qs)  
  end  
  θ .= θ′ 
  nothing
end