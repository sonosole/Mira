export meannorm

"""
    meannorm(x::Variable{T}; dims::IntOrDims{N}=1) where {T,N}
Return `(x .- μ)` of which μ = mean(x;dims)
# Gradient Dependencies
          ┌─────────────────────────┐
          │                         ▼
      ┌───┴───┐      ┌───┐      ┌───────┐        ┌───┐
      │ ──x── ├─────►│ μ │─────►│ ──y── ├──•••──►│ l │
      └───────┘      └───┘      └───────┘        └───┘
# Gradient Calculations
    μ  = ∑(xⱼ) / m
    yⱼ = xⱼ - μ
    ∂l∂xᵢ = ∂l∂yᵢ * ∂yᵢ∂xᵢ + ∑(∂l∂yⱼ * ∂yⱼ∂μ) * ∂μ∂xᵢ
          = ∂l∂yᵢ - ∑(∂l∂yᵢ) / m
"""
function meannorm(x::Variable{T}; dims::IntOrDims{N}=1) where {T,N}
    μ = mean(ᵛ(x); dims)
    y = Variable{T}(ᵛ(x) .- μ, x.backprop)

    if y.backprop
        m⁻¹ = eltype(x)(1 / prod(size(x, i) for i in dims))
        y.backward = function ∇meannorm()
            if need2computeδ!(x)
                ∂l∂y = δ(y)
                x ← ∂l∂y .- sum(∂l∂y; dims) .* m⁻¹
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function meannorm(x::AbstractArray; dims::IntOrDims{N}=1) where N
    return x .- mean(x; dims)
end
