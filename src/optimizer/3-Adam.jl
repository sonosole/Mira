export Adam

"""
    Adam(::Vector{XVariable}; lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, L1decay=0.001, L2decay=0.01)

Implements Adam algorithm. Refer to `Adam: A Method for Stochastic Optimization`.
"""
mutable struct Adam <: Optimizer
    xparams::Vector{XVariable}
    w1::Vector
    w2::Vector
    lr::AbstractFloat
    b1::AbstractFloat
    b2::AbstractFloat
    ϵ::AbstractFloat
    t::UInt
    b1t::AbstractFloat
    b2t::AbstractFloat
    L1decay::AbstractFloat
    L2decay::AbstractFloat
    name::String
    function Adam(xparams::Vector{XVariable}; lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, L1decay=0.001, L2decay=0.01)
        num = length(xparams)
        w1  = Vector(undef,num)
        w2  = Vector(undef,num)
        for i = 1:num
            c , θ = xparams[i]
            w1[i] = Zeros(typeof(ᵛ(θ)), θ.shape)
            w2[i] = Zeros(typeof(ᵛ(θ)), θ.shape)
        end
        new(xparams,w1,w2,lr, b1, b2, eps, 0, 1.0, 1.0, L1decay, L2decay, "Adam")
    end
end


function Base.show(io::IO, O::Adam)
    print("Adam(lr=$(O.lr), β₁=$(O.b1), β₂=$(O.b2), ϵ=$(O.ϵ), L1decay=$(O.L1decay), L2decay=$(O.L2decay))");
end



function update!(O::Adam;
                 clipfn::Function=LPInfNormClip,
                 clipvalue::Real=10.0,
                 applyL1::Function=decay_by_L₁,
                 applyL2::Function=decay_by_L₂)
    w₁ = O.w1
    w₂ = O.w2
    lr = O.lr
    b₁ = O.b1
    b₂ = O.b2
    ϵ  = O.ϵ
    λ₁ = O.L1decay
    λ₂ = O.L2decay
    O.t   += 1
    O.b1t *= b₁
    O.b2t *= b₂
    b₁ᵗ = O.b1t
    b₂ᵗ = O.b2t

    Threads.@threads for i = 1:length(O.xparams)
        c , θ = O.xparams[i]
        isnothing(δ(θ)) && continue
        setNanInfZero!(δ(θ))
        μ = - sqrt(1-b₂ᵗ) / (1-b₁ᵗ) * lr
        ∇ = clipfn(δ(θ), clipvalue)
        𝒗 = ᵛ(θ)
        @. w₁[i] = b₁ * w₁[i] + (1-b₁) * ∇
        @. w₂[i] = b₂ * w₂[i] + (1-b₂) * ∇ * ∇

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

        @. 𝒗 += μ * w₁[i] / sqrt(w₂[i] + ϵ)
    end
end
