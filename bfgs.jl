include("utils.jl")

mutable struct BFGS <: DescentMethod
  Q::Array{Float64, 2}
  BFGS() = new()
end

function init!(M::BFGS, θ)  
  p = length(θ)  
  # initiated with an identity matrix  
  M.Q = Matrix(1.0I, p, p)  
  return M 
end

function step!(M::BFGS, x, f, ∇f)  
  Q = M.Q # Hessian approximate
  x′  = ∇f(x) # gradient 
  d = -Q*x′ # direction
  ϕ  = α ->  f(x + α*d)
  ϕ′ = α -> ∇f(x + α*d)⋅d  
  α  = line_search(ϕ, ϕ′, d) # minimum of one dimen. function ϕ for stepsize α
  δ = α*d
  x .= x + δ # x
  γ  = ∇f(x) - x′
  Q .= Q - (δ*γ'*Q+Q*γ*δ')/(δ'*γ) +(1.0 + (γ'*Q*γ)/(δ'*γ)) *(δ*δ')/(δ'*γ)  
  return x 
end
