include("utils.jl")

mutable struct BFGS <: DescentMethod
  Q::Array{Float32, 2}
  BFGS() = new()
end

function init!(M::BFGS, θ::Array{Float32})  
  p = length(θ)  
  # initiated with an identity matrix  
  M.Q = Matrix(1.0I, p, p)  
  return M 
end

function step!(M::BFGS, θ::Array{Float32}, f, ∇f)  
  Q = M.Q # Hessian approximate
  θ′  = ∇f(θ) # gradient 
  d = -Q*θ′ # direction
  ϕ  = α -> f(θ + α*d)
  ϕ′ = α -> ∇f(θ + α*d)⋅d  
  α  = line_search(ϕ, ϕ′, d) # minimum of one dimen. function ϕ for stepsize α
  δ = α*d
  θ .= θ + δ # x
  γ  = ∇f(θ) - θ′
  Q .= Q - (δ*γ'*Q+Q*γ*δ')/(δ'*γ) +(1.0 + (γ'*Q*γ)/(δ'*γ)) *(δ*δ')/(δ'*γ)  
  nothing
end
