function Base.maximum(x::Variable{T}; dims::Union{Int,NTuple{N,Int}}=1) where {T,N}
    y = Variable{T}(maximum(ᵛ(x), dims=dims), x.backprop)
    if y.backprop
        mask = ᵛ(x) .== ᵛ(y)
        y.backward = function ∇maximum()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* mask
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function Base.minimum(x::Variable{T}; dims::Union{Int,NTuple{N,Int}}=1) where {T,N}
    y = Variable{T}(minimum(ᵛ(x), dims=dims), x.backprop)
    if y.backprop
        mask = ᵛ(x) .== ᵛ(y)
        y.backward = function ∇minimum()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* mask
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function Base.sum(x::Variable{T}; dims::Union{Int,NTuple{N,Int}}=1) where {T,N}
    y = Variable{T}(sum(ᵛ(x), dims=dims), x.backprop)
    if y.backprop
        y.backward = function ∇sum()
            if need2computeδ!(x)
                δ(x) .+= δ(y)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function mean(x::Variable{T}; dims::Union{Int,NTuple{N,Int}}=1) where {T,N}
    n = eltype(x)(1) / prod(size(x, i) for i in dims)
    μ = Variable{T}(sum(ᵛ(x), dims=dims) .* n, x.backprop)
    if μ.backprop
        μ.backward = function ∇mean()
            if need2computeδ!(x)
                δ(x) .+= δ(μ) .* n
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
                δ(x) .+= δ(y) .* mask
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


function linearpool(x::Variable{T}; dims::Union{Int,NTuple{N,Int}}=1) where {T,N}
    Σxᵢ² = sum(ᵛ(x) .* ᵛ(x), dims=dims)     # Σ xᵢ·xᵢ
    Σxᵢ  = sum(ᵛ(x),         dims=dims)     # Σ xᵢ
    y    = Variable{T}(Σxᵢ² ./ Σxᵢ, x.backprop)
    if y.backprop
        𝟚 = eltype(x)(2.0f0)
        y.backward = function ∇linearpool()
            if need2computeδ!(x)
                δ(x) .+= (𝟚 .* ᵛ(x) .- ᵛ(y)) ./ Σxᵢ .* δ(y)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


"""
    linearpool(x::AbstractArray; dims=1) -> y

    y = (Σᵢ xᵢ .* xᵢ) ./ Σᵢ xᵢ
"""
function linearpool(x::AbstractArray; dims::Union{Int,NTuple{N,Int}}=1) where N
    return sum(x .* x, dims=dims) ./ sum(x, dims=dims)
end


function exppool(x::Variable{T}; dims::Union{Int,NTuple{N,Int}}=1) where {T,N}
    eˣ  = exp.(ᵛ(x))
    Σeˣⁱxᵢ = sum(eˣ .* ᵛ(x), dims=dims)   # Σ exp(xᵢ)·xᵢ
    Σeˣⁱ = sum(eˣ, dims=dims)             # Σ exp(xᵢ)
    y  = Variable{T}(Σeˣⁱxᵢ ./ Σeˣⁱ, x.backprop)
    if y.backprop
        𝟙 = eltype(x)(1.0f0)
        y.backward = function ∇exppool()
            if need2computeδ!(x)
                δ(x) .+= eˣ ./ Σeˣⁱ .* (𝟙 .+ ᵛ(x) .- ᵛ(y)) .* δ(y)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


"""
    exppool(x::AbstractArray; dims=1) -> y

    y = (Σᵢ exp.(xᵢ) .* xᵢ) ./ Σᵢ exp.(xᵢ)
"""
function exppool(x::AbstractArray; dims::Union{Int,NTuple{N,Int}}=1) where N
    e = exp.(x)
    return sum(e .* x, dims=dims) ./ sum(e, dims=dims)
end



export mean
export maxmin
export linearpool
export exppool
