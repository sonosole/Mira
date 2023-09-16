export mean
export var
export std
export maxmin
export linearpool
export powerpool
export exppool


function Base.maximum(x::Variable{T}; dims::IntOrDims{N}=1) where {T,N}
    v, i = findmax(ᵛ(x); dims)
    y = Variable{T}(v, x.backprop)
    if y.backprop
        y.backward = function ∇maximum()
            if needgrad(x)
                zerodelta(x)
                ᵟ(x)[i] .+= δ(y)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function Base.minimum(x::Variable{T}; dims::IntOrDims{N}=1) where {T,N}
    v, i = findmin(ᵛ(x); dims)
    y = Variable{T}(v, x.backprop)
    if y.backprop
        y.backward = function ∇minimum()
            if needgrad(x)
                zerodelta(x)
                ᵟ(x)[i] .+= δ(y)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function Base.sum(x::Variable{T}; dims::IntOrDims{N}=1) where {T,N}
    y = Variable{T}(sum(ᵛ(x); dims), x.backprop)
    if y.backprop
        y.backward = function ∇sum()
            if needgrad(x)
                x ← δ(y) .+ zero(x)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function mean(x::Variable{T}; dims::IntOrDims{N}=1) where {T,N}
    m⁻¹ = eltype(x)(1 / prod(size(x, i) for i in dims))
    μ   = Variable{T}(sum(ᵛ(x); dims) .* m⁻¹, x.backprop)
    if μ.backprop
        μ.backward = function ∇mean()
            if needgrad(x)
                x ← δ(μ) .* m⁻¹ .+ zero(x)
            end
            ifNotKeepδThenFreeδ!(μ)
        end
        addchild(μ, x)
    end
    return μ
end

"""
    var(x::Variable{T}; dims::IntOrDims{N}=1) where {T,N}
Variance of `x`
# Gradient dependencies
          ┌──────────────────────┐
          │                      ▼
        ┌─┴─┐       ┌───┐      ┌─┴─┐
        │ x ├──────►│ μ │      │ σ²├── ••• ──► l
        └───┘       └───┘      └───┘

             ∂l     ∂σ²     ∂l    ∂σ²    ∂μ    ∂l     ∂σ²
    ∂l/∂x =  ─── • ────  +  ─── • ─── • ──── = ─── • ────
             ∂σ²    ∂x      ∂σ²   ∂μ     ∂x    ∂σ²    ∂x
"""
function var(x::Variable{T}; dims::IntOrDims{N}=1) where {T,N}
    μ  = mean(ᵛ(x); dims)
    v  =  var(ᵛ(x); dims, mean=μ, corrected=false)
    σ² = Variable{T}(v, x.backprop)

    if σ².backprop
        𝟐𝐦⁻¹ = eltype(x)(2 / prod(size(x, i) for i in dims))
        σ².backward = function ∇mean()
            if needgrad(x)
                x ← δ(σ²) .* (ᵛ(x) .- μ) .* 𝟐𝐦⁻¹
            end
            ifNotKeepδThenFreeδ!(σ²)
        end
        addchild(σ², x)
    end
    return σ²
end

function std(x::Variable{T}; dims::IntOrDims{N}=1) where {T,N}
    return sqrt(var(x, dims=dims))
end

function maxmin(x::Variable{T}; dims1::IntOrDims{N1}, dims2::IntOrDims{N2}) where {T,N1,N2}
    return minimum( maximum(x, dims=dims1), dims=dims2)
end

function maxmin(x::AbstractArray; dims1::IntOrDims{N1}, dims2::IntOrDims{N2}) where {N1,N2}
    return minimum( maximum(x, dims=dims1), dims=dims2)
end


function Base.minmax(x::Variable{T}; dims1::IntOrDims{N1}, dims2::IntOrDims{N2}) where {T,N1,N2}
    return maxmin(x; dims1=dims2, dims2=dims1)
end

function Base.minmax(x::AbstractArray; dims1::IntOrDims{N1}, dims2::IntOrDims{N2}) where {N1,N2}
    return maximum(minimum(x, dims=dims1), dims=dims2)
end


"""
    linearpool(x::Variable; dims=2) -> y::Variable

y[k] = (Σᵢ x[k,i] * x[k,i]) / Σᵢ x[k,i], i is the indices of other dims
"""
function linearpool(x::Variable{T}; dims::IntOrDims{N}=2) where {T,N}
    Σxᵢ² = sum(ᵛ(x) .* ᵛ(x), dims=dims)     # Σ xᵢ·xᵢ
    Σxᵢ  = sum(ᵛ(x),         dims=dims)     # Σ xᵢ
    y    = Variable{T}(Σxᵢ² ./ Σxᵢ, x.backprop)
    if y.backprop
        𝟚 = eltype(x)(2)
        y.backward = function ∇linearpool()
            if needgrad(x)
                x ← (𝟚 .* ᵛ(x) .- ᵛ(y)) ./ Σxᵢ .* δ(y) .+ zero(x)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


"""
    linearpool(x::AbstractArray; dims=2) -> y::AbstractArray

    y[k] = (Σᵢ x[k,i] * x[k,i]) / Σᵢ x[k,i], i is the indices of other dims
"""
function linearpool(x::AbstractArray; dims::IntOrDims{N}=2) where N
    return sum(x .* x, dims=dims) ./ sum(x, dims=dims)
end


"""
    exppool(x::Variable; dims=2) -> y::Variable

    y[k] = (Σᵢ exp(x[k,i]) * x[k,i]) / Σᵢ exp(x[k,i]), i is the indices of other dims
"""
function exppool(x::Variable{T}; dims::IntOrDims{N}=2) where {T,N}
    eˣ  = exp.(ᵛ(x))
    Σeˣⁱxᵢ = sum(eˣ .* ᵛ(x), dims=dims)   # Σ exp(xᵢ)·xᵢ
    Σeˣⁱ = sum(eˣ, dims=dims)             # Σ exp(xᵢ)
    y  = Variable{T}(Σeˣⁱxᵢ ./ Σeˣⁱ, x.backprop)
    if y.backprop
        l = eltype(x)(1)
        y.backward = function ∇exppool()
            if needgrad(x)
                x ← eˣ ./ Σeˣⁱ .* (l .+ ᵛ(x) .- ᵛ(y)) .* δ(y) .+ zero(x)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


"""
    exppool(x::AbstractArray; dims=2) -> y::AbstractArray

    y[k] = (Σᵢ exp(x[k,i]) * x[k,i]) / Σᵢ exp(x[k,i]), i is the indices of other dims
"""
function exppool(x::AbstractArray; dims::IntOrDims{N}=2) where N
    eˣ = exp.(x)
    return sum(eˣ .* x, dims=dims) ./ sum(eˣ, dims=dims)
end


"""
    powerpool(x::Variable, n::Real=3; dims=2) -> y::Variable

    y =  (Σxᵢⁿ · xᵢ) / Σxᵢⁿ, i is the indices for aggregate
"""
function powerpool(x::Variable{T}, n::Real=3; dims::IntOrDims{N}=2) where {T,N}
    xᵢⁿ    = ᵛ(x) .^ n
    Σxᵢⁿ⁺¹ = sum(xᵢⁿ .* ᵛ(x), dims=dims)    # Σxᵢⁿ · xᵢ
    Σxᵢⁿ   = sum(xᵢⁿ,         dims=dims)    # Σxᵢⁿ
    y = Variable{T}(Σxᵢⁿ⁺¹ ./ Σxᵢⁿ, x.backprop)
    if y.backprop
        l = eltype(x)(1.0f0)
        xᵢⁿ⁻¹ = ᵛ(x) .^ (n-1)
        y.backward = function ∇powerpool()
            if needgrad(x)
                x ← ((n+l) .* xᵢⁿ .- n .* xᵢⁿ⁻¹ .* ᵛ(y)) ./ Σxᵢⁿ .* δ(y) .+ zero(x)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


"""
    powerpool(x::AbstractArray, n::Real=3; dims=2) -> y::AbstractArray

    y =  (Σxᵢⁿ · xᵢ) / Σxᵢⁿ, i is the indices for aggregate
"""
function powerpool(x::AbstractArray, n::Real=3; dims::IntOrDims{N}=2) where N
    xᵢⁿ    = x .^ n
    Σxᵢⁿ⁺¹ = sum(xᵢⁿ .* ᵛ(x), dims=dims)    # Σxᵢⁿ · xᵢ
    Σxᵢⁿ   = sum(xᵢⁿ,         dims=dims)    # Σxᵢⁿ
    return Σxᵢⁿ⁺¹ ./ Σxᵢⁿ
end


function nilnorm(x; dims::IntOrDims=1)
    return x
end
