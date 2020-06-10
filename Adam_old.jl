include("utils.jl")

mutable struct Adam_old <: DescentMethod
  α # learning rate  
  ε # small value  
  v # sum gradient  
  s # sum of squared gradient  
  γs # gradient decay  
  γv # momentum decay  
  k # step counter  
  Adam_old() = new() # new uninitialized structure 
end 

function init!(M::Adam_old, θ; α=0.001, γv=0.9, γs=0.999, ε=1e-4)
  M.α = α  
  M.ε = ε  
  M.γs = γs  
  M.γv = γv  
  M.s = zero(θ)  
  M.v = zero(θ)  
  M.k = 0  
  return M 
end

function step!(M::Adam_old, θ, f, ∇f)
   γs, γv  = M.γs, M.γv  
   α, ε, k = M.α, M.ε, M.k  
   s, v, g = M.s, M.v, ∇f(θ)  
   v[:] = γv*v + (1.0 - γv) * g  
   s[:] = γs*s + (1.0 - γs) * g .* g  
   M.k = k += 1  
   v̂ = v ./ (1.0 - γv^k)  
   ŝ = s ./ (1.0 - γs^k)  
   return θ - α*v̂ ./ (sqrt.(ŝ) .+ ε) 
end
