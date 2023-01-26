export Loss
export cost


# Internal function
function _sum(x::Variable{T}) where T
    y = Variable{T}([sum(ᵛ(x))], x.backprop)
    if y.backprop
        y.backward = function _sumBackward()
            if need2computeδ!(x)
                δ(x) .+= δ(y)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


# Internal function
function _mean(x::Variable{T}) where T
    n = eltype(x)(1) / prod(size(x))
    μ = Variable{T}([sum(ᵛ(x)) * n], x.backprop)
    if μ.backprop
        μ.backward = function _meanBackward()
            if need2computeδ!(x)
                δ(x) .+= δ(μ) .* n
            end
            ifNotKeepδThenFreeδ!(μ);
        end
        addchild(μ, x)
    end
    return μ
end


"""
    Loss(x::Variable{T}; reduction::String="sum") -> y::Variable{T}

Sums or takes mean over all elements in `value` of `x` as the loss `Variable`, i.e. ⤦\n
+ `y = Variable{T}([sum(ᵛ(x))], x.backprop)`, if reduction="sum"
+ `y = Variable{T}([sum(ᵛ(x))/length(x)], x.backprop)`, if reduction="mean"
This is very convenient for mutiple loss training. e.g. ⤦\n
    totalLoss = β*loss(mse(x₁, ̂x₁)) + (1 - β)*loss(crossEntropy(y₁, ̂y₁))
or in a lazy way:\n
    totalLoss = β*mseLoss(x₁, ̂x₁) + (1 - β)*crossEntropyLoss(y₁, ̂y₁)
where `β` is weight of mseLoss function.
"""
function Loss(x::Variable{T}; reduction::String="sum") where T
    by = lowercase(reduction)
    by=="sum" && return _sum(x)
    by=="mean" && return _mean(x)
    @error "wrong reduction method" reduction=="sum" || reduction=="mean"
end


function cost(x::Variable)
    @assert length(x)==1
    return Array(x.value)[1]
end




function _sum(x::AbstractArray)
    return sum(x)
end

function _mean(x::AbstractArray)
    return sum(x) / length(x)
end


"""
    Loss(x::AbstractArray; reduction::String="sum") -> y::AbstractArray

Sums or takes mean over all elements in `value` of `x` as the loss `Variable`, i.e. ⤦\n
+ `y = sum(x)`, if reduction="sum"
+ `y = sum(x)/length(x)`, if reduction="mean"
"""
function Loss(x::AbstractArray; reduction::String="sum")
    by = lowercase(reduction)
    by=="sum" && return _sum(x)
    by=="mean" && return _mean(x)
    @error "wrong reduction method" reduction=="sum" || reduction=="mean"
end


include("./1-criterion-regression.jl")
include("./1-criterion-probabilistic.jl")
