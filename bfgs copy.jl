include("utils.jl")

mutable struct BFGS  
  Q # approximation of    
  # Hessian matrix inverse  
  BFGS() = new() 
end 
function bfgs_init!(M::BFGS, θ)  
  p = length(θ)  
  # initiated with an identity matrix  
  M.Q = Matrix(1.0I, p, p)  
  return M 
end
function bfgs_step!(M::BFGS, f, ∇f, θ)  
  Q  = M.Q  
  g  = ∇f(θ)  
  d  = -Q*g  
  ϕ  = α ->  f(θ + α*d)  
  ϕ′ = α -> ∇f(θ + α*d)⋅d  
  α  = line_search(ϕ, ϕ′, d)  
  θ′ = θ + α*d  
  g′ = ∇f(θ′)  
  δ  = θ′ - θ  
  γ  = g′ - g  
  Q[:]= Q - (δ*γ'*Q+Q*γ*δ')/(δ'*γ) +(1.0 + (γ'*Q*γ)/(δ'*γ)) *(δ*δ')/(δ'*γ)  
  return θ′ 
end
