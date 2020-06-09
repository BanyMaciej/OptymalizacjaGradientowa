mutable struct Adam{T<:AbstractFloat}
  α::T # learning rate  
  ε::T # small value  
  v # sum gradient  
  s # sum of squared gradient  
  γs # gradient decay  
  γv # momentum decay  
  k::Int   # step counter  
  Adam() = new{Float64}() # new uninitialized structure 
end 

function init!(M::Adam, θ; α=0.001, γv=0.9, γs=0.999, ε=1e-4)
  M.α = α  
  M.ε = ε  
  M.γs = γs  
  M.γv = γv  
  M.s = zero(θ)  
  M.v = zero(θ)  
  M.k = 0  
  return M 
end

function updateGradient!(v, γv, g) 
  v .= γv*v + (1.0 - γv) * g 
  nothing
end

function updateGradientSquaredSum!(s, γs, g)
  s .= γs*s + (1.0 - γs) * g .* g
  nothing
end

@fastmath function step!(M::Adam, θ, f, ∇f)
  γs, γv  = M.γs, M.γv  
  α, ε, k = M.α, M.ε, M.k  
  s, v, g = M.s, M.v, ∇f(θ) 
  updateGradient!(v, γv, g)
  updateGradientSquaredSum!(s, γs, g)
  M.k = k += 1  
  v̂ = v ./ (1.0 - γv^k)  
  ŝ = s ./ (1.0 - γs^k)  
  θ .= θ - α*v̂ ./ (sqrt.(ŝ) .+ ε) 
  nothing
end
