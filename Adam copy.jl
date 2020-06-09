mutable struct Adam_temp
  α::Float64
  γv::Float64
  γs::Float64
  ε::Float64
  state::IdDict
end
Adam_temp(α = 0.001, γv = 0.9, γs = 0.999, ε=1e-8) = Adam_temp(α, γv, γs, ε, IdDict())

function step!(o::Adam_temp, x, f::Function, ∇f::Function)
  x′ = ∇f(x)
  α, γv, γs = o.α, o.γv, o.γs
  mt, vt, γv_p, γs_p = get!(o.state, x, (zero(x), zero(x), γv, γs))
  @. mt = γv * mt + (1 - γv) * x′
  @. vt = γs * vt + (1 - γs) * x′^2
  @. x′ = mt / (1 - γv_p) / (√(vt / (1 - γs_p)) + o.ε) * α
  o.state[x] = (mt, vt, γv_p*γv, γs_p*γs)
  x .-= x′
  return x
end