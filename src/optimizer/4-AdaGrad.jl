export AdaGrad

"""
    AdaGrad(::Vector{XVariable}; lr=1e-2, eps=1e-10, L1decay=0.001, L2decay=0.01)

Implements Adagrad algorithm. Refer to `Adaptive Subgradient Methods for Online Learning and Stochastic`
"""
mutable struct AdaGrad <: Optimizer
    xparams::Vector{XVariable}
    w::Vector
    lr::AbstractFloat
    ϵ::AbstractFloat
    L1decay::AbstractFloat
    L2decay::AbstractFloat
    name::String
    function AdaGrad(xparams::Vector{XVariable}; lr=1e-2, eps=1e-10, L1decay=0.001, L2decay=0.01)
        n = length(xparams)
        w = Vector(undef,n)
        for i = 1:n
            c , θ = xparams[i]
            w[i] = Zeros(typeof(ᵛ(θ)), θ.shape)
        end
        new(xparams, w, lr, eps, L1decay, L2decay, "AdaGrad")
    end
end


function Base.show(io::IO, O::AdaGrad)
    print("AdaGrad(lr=$(O.lr), ϵ=$(O.ϵ), L1decay=$(O.L1decay), L2decay=$(O.L2decay))")
end


function update!(O::AdaGrad;
                 clipfn::Function=LPInfNormClip,
                 clipvalue::Real=10.0,
                 applyL1::Function=decay_by_L₁,
                 applyL2::Function=decay_by_L₂)
    w = O.w
    μ = - O.lr
    ϵ = O.ϵ
    λ₁ = O.L1decay
    λ₂ = O.L2decay

    Threads.@threads for i = 1:length(O.xparams)
        c , θ = O.xparams[i]
        isnothing(δ(θ)) && continue
        setNanInfZero!(δ(θ))
        ∇ = clipfn(δ(θ), clipvalue)
        𝒗 = ᵛ(θ)
        @. w[i] += ∇ * ∇

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

        @. 𝒗 += μ / (sqrt(w[i]) + ϵ) * ∇
    end
end
