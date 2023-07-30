const IntOrDims{N} = Union{Int, Dims{N}} where N

function Base.maximum(x::Variable{T}; dims::IntOrDims{N}=1) where {T,N}
    y = Variable{T}(maximum(ᵛ(x), dims=dims), x.backprop)
    if y.backprop
        mask = ᵛ(x) .== ᵛ(y)
        y.backward = function ∇maximum()
            if need2computeδ!(x)
                x ← δ(y) .* mask .+ zero(x)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function Base.minimum(x::Variable{T}; dims::IntOrDims{N}=1) where {T,N}
    y = Variable{T}(minimum(ᵛ(x), dims=dims), x.backprop)
    if y.backprop
        mask = ᵛ(x) .== ᵛ(y)
        y.backward = function ∇minimum()
            if need2computeδ!(x)
                x ← δ(y) .* mask .+ zero(x)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function Base.sum(x::Variable{T}; dims::IntOrDims{N}=1) where {T,N}
    y = Variable{T}(sum(ᵛ(x), dims=dims), x.backprop)
    if y.backprop
        y.backward = function ∇sum()
            if need2computeδ!(x)
                x ← δ(y) .+ zero(x)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function mean(x::Variable{T}; dims::IntOrDims{N}=1) where {T,N}
    n = eltype(x)(1) / prod(size(x, i) for i in dims)
    μ = Variable{T}(sum(ᵛ(x), dims=dims) .* n, x.backprop)
    if μ.backprop
        μ.backward = function ∇mean()
            if need2computeδ!(x)
                x ← δ(μ) .* n .+ zero(x)
            end
            ifNotKeepδThenFreeδ!(μ)
        end
        addchild(μ, x)
    end
    return μ
end


function maxmin(x::Variable{T}; dims1::Int, dims2::Int) where T
    t = minimum(maximum(ᵛ(x), dims=dims1), dims=dims2)
    y = Variable{T}(t, x.backprop)
    if y.backprop
        mask = ᵛ(x) .== ᵛ(y)
        y.backward = function ∇maxmin()
            if need2computeδ!(x)
                x ← δ(y) .* mask .+ zero(x)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function maxmin(x::AbstractArray; dims1::Int, dims2::Int)
    return minimum( maximum(x, dims=dims1), dims=dims2)
end

function Base.minmax(x::Variable{T}; dims1::Int, dims2::Int) where T
    return maxmin(x; dims1=dims2, dims2=dims1)
end


function Base.minmax(x::AbstractArray; dims1::Int, dims2::Int)
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
        𝟚 = eltype(x)(2.0f0)
        y.backward = function ∇linearpool()
            if need2computeδ!(x)
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
        𝟙 = eltype(x)(1.0f0)
        y.backward = function ∇exppool()
            if need2computeδ!(x)
                x ← eˣ ./ Σeˣⁱ .* (𝟙 .+ ᵛ(x) .- ᵛ(y)) .* δ(y) .+ zero(x)
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
    e = exp.(x)
    return sum(e .* x, dims=dims) ./ sum(e, dims=dims)
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
            if need2computeδ!(x)
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


export mean
export maxmin
export linearpool
export powerpool
export exppool
