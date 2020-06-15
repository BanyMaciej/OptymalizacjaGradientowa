include("utils.jl")

mutable struct Adam <: DescentMethod
  α::Float32 # learning rate  
  ε::Float32 # small value  
  v::Array{Float32} # sum gradient  
  s::Array{Float32} # sum of squared gradient  
  γs::Float32 # gradient decay  
  γv::Float32 # momentum decay  
  k::Int   # step counter  
  Adam() = new() # new uninitialized structure 
end 

function init!(M::Adam, θ::Array{Float32}; α=0.001, γv=0.9, γs=0.999, ε=1e-4)
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

function step!(M::Adam, θ::Array{Float32}, f, ∇f)
  γs, γv  = M.γs, M.γv  
  α, ε, k = M.α, M.ε, M.k  
  s, v, g = M.s, M.v, ∇f(θ) 
  updateGradient!(v, γv, g)
  updateGradientSquaredSum!(s, γs, g)
  M.k = k += 1  
  v̂ = v ./ (1.0 - γv^k)  
  ŝ = s ./ (1.0 - γs^k)  
  θ .-= α*v̂ ./ (sqrt.(ŝ) .+ ε) 
  nothing
end
