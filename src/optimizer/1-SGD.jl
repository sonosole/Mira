export SGD

"""
    SGD(::Vector{XVariable}; lr=1e-4, L1decay=0.001, L2decay=0.01)

Implements stochastic gradient descent
"""
mutable struct SGD <: Optimizer
    xparams::Vector{XVariable}
    lr::AbstractFloat
    L1decay::AbstractFloat
    L2decay::AbstractFloat
    name::String
    function SGD(xparams::Vector{XVariable}; lr=1e-4, L1decay=0.001, L2decay=0.01)
        new(xparams, lr, L1decay, L2decay, "SGD")
    end
end

# pretty printing
function Base.show(io::IO, O::SGD)
    print("SGD(lr=$(O.lr), L1decay=$(O.L1decay), L2decay=$(O.L2decay))")
end


function update!(O::SGD;
                 clipfn::Function=LPInfNormClip,
                 clipvalue::Real=10.0,
                 applyL1::Function=decay_by_L₁,
                 applyL2::Function=decay_by_L₂)
    μ = - O.lr
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

        @. 𝒗 += μ * ∇
    end
end
