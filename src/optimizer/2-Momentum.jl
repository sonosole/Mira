export Momentum

"""
    Momentum(::Vector{XVariable}; lr=1e-4, inertia=0.9, L1decay=0.001, L2decay=0.01)

Implements stochastic gradient descent with momentum
"""
mutable struct Momentum <: Optimizer
    xparams::Vector{XVariable}
    v::Vector
    lr::AbstractFloat
    inertia::AbstractFloat
    L1decay::AbstractFloat
    L2decay::AbstractFloat
    name::String
    function Momentum(xparams::Vector{XVariable}; lr=1e-4, inertia=0.9, L1decay=0.001, L2decay=0.01)
        num = length(xparams)
        vel = Vector(undef,num)
        for i = 1:num
            c , θ = xparams[i]
            vel[i] = Zeros(typeof(ᵛ(θ)), θ.shape)
        end
        new(xparams, vel, lr, inertia, L1decay, L2decay, "Momentum")
    end
end


function Base.show(io::IO, O::Momentum)
    print("Momentum(lr=$(O.lr), inertia=$(O.inertia), L1decay=$(O.L1decay), L2decay=$(O.L2decay))")
end


function update!(O::Momentum;
                 clipfn::Function=LPInfNormClip,
                 clipvalue::Real=10.0,
                 applyL1::Function=decay_by_L₁,
                 applyL2::Function=decay_by_L₂)
    v = O.v
    μ = - O.lr
    ρ = O.inertia
    λ₁ = O.L1decay
    λ₂ = O.L2decay

    Threads.@threads for i = 1:length(O.xparams)
        c , θ = O.xparams[i]
        isnothing(δ(θ)) && continue
        setNanInfZero!(δ(θ))
        ∇ = clipfn(δ(θ), clipvalue)
        𝒗 = ᵛ(θ)

        L₁ = applyL1(c) && λ₁ ≠ 0   # whether do L₁ regularization
        L₂ = applyL2(c) && λ₂ ≠ 0   # whether do L₂ regularization

        if !L₁ && L₂
            @. 𝒗 += μ * λ₂ * 𝒗
        end
        if L₁ && !L₂
            @. 𝒗 += μ * λ₁ * sign(𝒗)
        end
        if L₁ && L₂
            @. 𝒗 += μ * λ₁ * sign(𝒗) + μ * λ₂ * 𝒗
        end

        @. v[i] = ρ * v[i] + ∇
        @. 𝒗 += μ * v[i]
    end
end
