export znorm

"""
    znorm(x::Variable{T}; dims::IntOrDims{N}=1, eps::Real=1e-38) where {T,N}
Return `(x .- μ) ./ σ` of which:
    μ = mean(x;dims),
    σ =  std(x;dims)
# Gradient Dependencies
          ┌───────────5──────────┐
          │                      ▼
      ┌───┴───┐      ┌───┐     ┌────┐      ┌───────┐        ┌───┐
      │ ──x── ├──4──►│ μ │     │ σ² ├──3──►│ ──y── ├──•••──►│ l │
      └───┬───┘      └─┬─┘     └────┘      └───────┘        └───┘
          │            │                    ▲     ▲
          │            └───────────2────────┘     │
          └────────────────────────1──────────────┘
# Gradient Calculations
    x̄ᵢ     = xᵢ - μ
    x̌ᵢ     = x̄ᵢ * σ¯¹
    yᵢ     = x̌ᵢ

    ∂yᵢ∂xᵢ =  1/σ               # 1
    ∂yᵢ∂μ  = -1/σ               # 2
    ∂yᵢ∂σ² = -1/2 * σ¯³ * x̄ᵢ    # 3
    ∂μ∂xᵢ  = 1/m                # 4
    ∂σ²∂xᵢ = 2/m * x̄ᵢ           # 5
    ∂l∂σ²  = ∑(∂l∂yᵢ * ∂yᵢ∂σ²)
    ∂l∂μ   = ∑(∂l∂yᵢ * ∂yᵢ∂μ)

    ∂l∂xᵢ = ∂l∂yᵢ * ∂yᵢ∂xᵢ  +  ∂l∂μ * ∂μ∂xᵢ              +  ∂l∂σ² * ∂σ²∂xᵢ
          = ∂l∂yᵢ * ∂yᵢ∂xᵢ  +  ∑(∂l∂yᵢ * ∂yᵢ∂μ) * ∂μ∂xᵢ  +  ∑(∂l∂yᵢ * ∂yᵢ∂σ²) * ∂σ²∂xᵢ
          = ∂l∂yᵢ * σ¯¹     -  ∑(∂l∂yᵢ * σ¯¹) * (1/m)    +  ∑(∂l∂yᵢ * (-1/2 * σ¯³) * x̄ᵢ) * (2/m) * x̄ᵢ
          = (∂l∂yᵢ          -  ∑(∂l∂yᵢ) * (1/m)          +  ∑(∂l∂yᵢ * (-1/2 * σ¯²) * x̄ᵢ) * (2/m) * x̄ᵢ) * σ¯¹
          = (m * ∂l∂yᵢ      -  ∑(∂l∂yᵢ)                  -  ∑(∂l∂yᵢ * x̄ᵢ * σ¯¹) * x̄ᵢ * σ¯¹) * (σ¯¹ * m¯¹)
          = (m * ∂l∂yᵢ      -  ∑(∂l∂yᵢ)                  -  ∑(∂l∂yᵢ * yᵢ)       * yᵢ)       * (σ¯¹ * m¯¹)
"""
function znorm(x::Variable{T}; dims::IntOrDims{N}=1, eps::Real=1e-38) where {T,N}
    D = eltype(x)
    l = D(1)
    ϵ = D(eps)
    m = D(prod(size(x, i) for i in dims))

    μ   = mean(ᵛ(x); dims)
    σ²  =  var(ᵛ(x); dims, corrected=false) .+ ϵ
    σ¯¹ = l ./ sqrt.(σ²)
    m¯¹ = l  / m
    x̌   = (ᵛ(x) .- μ) .* σ¯¹
    y   = Variable{T}(x̌, x.backprop)

    if y.backprop
        ∑(a::AbstractArray) = sum(a; dims=dims)
        y.backward = function ∇znorm()
            if need2computeδ!(x)
                ∂l∂y = δ(y)
                x ← (σ¯¹ .* m¯¹) .* (m .* ∂l∂y .- ∑(∂l∂y) .- x̌ .* ∑(∂l∂y .* x̌))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function znorm_mean_var(x::Variable{T}; dims::IntOrDims{N}=1, eps::Real=1e-38) where {T,N}
    D = eltype(x)
    l = D(1)
    ϵ = D(eps)
    m = D(prod(size(x, i) for i in dims))

    μ   = mean(ᵛ(x); dims)
    σ²  =  var(ᵛ(x); dims, corrected=false) .+ ϵ
    σ¯¹ = l ./ sqrt.(σ²)
    m¯¹ = l  / m
    x̌   = (ᵛ(x) .- μ) .* σ¯¹
    y   = Variable{T}(x̌, x.backprop)

    if y.backprop
        ∑(a::AbstractArray) = sum(a; dims=dims)
        y.backward = function ∇znorm()
            if need2computeδ!(x)
                ∂l∂y = δ(y)
                x ← (σ¯¹ .* m¯¹) .* (m .* ∂l∂y .- ∑(∂l∂y) .- x̌ .* ∑(∂l∂y .* x̌))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y, μ, σ²
end


function znorm(x::AbstractArray; dims::IntOrDims{N}=1, eps::Real=1e-38) where N
    l  = eltype(x)(1)
    ϵ  = eltype(x)(eps)
    μ  = mean(x; dims)
    σ² =  var(x; dims, mean=μ, corrected=false) .+ ϵ
    return @. (x - μ) * (l / sqrt(σ))
end


function znorm_mean_var(x::AbstractArray; dims::IntOrDims{N}=1, eps::Real=1e-38) where N
    l  = eltype(x)(1)
    ϵ  = eltype(x)(eps)
    μ  = mean(x; dims)
    σ² =  var(x; dims, mean=μ, corrected=false) .+ ϵ
    return @. (x - μ) * (l / sqrt(σ²)), μ, σ
end
