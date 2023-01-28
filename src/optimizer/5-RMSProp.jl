export RMSProp

"""
    RMSProp(::Vector{XVariable}; lr=1e-2, inertia=0.99, eps=1e-8, L1decay=0.001, L2decay=0.01)

Implements RMSProp algorithm.
"""
mutable struct RMSProp <: Optimizer
    xparams::Vector{XVariable}
    w::Vector
    lr::AbstractFloat
    ϵ::AbstractFloat
    inertia::AbstractFloat
    L1decay::AbstractFloat
    L2decay::AbstractFloat
    name::String
    function RMSProp(xparams::Vector{XVariable}; lr=1e-2, inertia=0.99, eps=1e-8, L1decay=0f0, L2decay=0f0)
        n = length(xparams)
        w = Vector(undef,n)
        for i = 1:n
            c , θ = xparams[i]
            w[i] = Zeros(typeof(ᵛ(θ)), θ.shape)
        end
        new(xparams, w, lr, eps, inertia, L1decay, L2decay, "RMSProp")
    end
end


function Base.show(io::IO, O::RMSProp)
    print("RMSProp(lr=$(O.lr), ϵ=$(O.ϵ), inertia=$(O.inertia), L1decay=$(O.L1decay), L2decay=$(O.L2decay))")
end


function update!(O::RMSProp;
                 clipfn::Function=LPInfNormClip,
                 clipvalue::Real=10.0,
                 applyL1::Function=decay_by_L₁,
                 applyL2::Function=decay_by_L₂)
    w = O.w
    ϵ = O.ϵ
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
        @. w[i] = ρ * w[i] + (1-ρ) * ∇ * ∇

        # regularization should be done before 𝒗 has been changed by ∇
        L₁ = applyL1(c) && λ₁ ≠ 0   # whether do L₁ regularization
        L₂ = applyL2(c) && λ₂ ≠ 0   # whether do L₂ regularization

        if !L₁ && L₂
            @. 𝒗 += (μ * λ₂) * 𝒗
        end
        if L₁ && !L₂
            @. 𝒗 += (μ * λ₁) * sign(𝒗)
        end
        if L₁ && L₂
            @. 𝒗 += (μ * λ₁) * sign(𝒗) + (μ * λ₂) * 𝒗
        end

        @. 𝒗 += μ / (sqrt(w[i])+ϵ) * ∇
    end
end
