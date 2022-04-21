## regression loss

export mae
export maeLoss
export L1Loss

export mse
export mseLoss
export L2Loss

export Lp, LpLoss


"""
    mae(x::Variable{T}, label::Variable{T}) -> y::Variable{T}

mean absolute error (mae) between each element in the input `x` and target `label`. Also called L1Loss. i.e. ⤦\n
    y = |xᵢ - lᵢ|
"""
function mae(x::Variable{T}, label::Variable{T}) where T
    @assert (x.shape == label.shape)
    backprop = (x.backprop || label.backprop)
    y = Variable{T}(abs.(ᵛ(x) - ᵛ(label)), backprop)
    if backprop
        y.backward = function maeBackward()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* sign.(ᵛ(y))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function mae(x::Variable{T}, label::AbstractArray) where T
    @assert x.shape == size(label)
    y = Variable{T}(abs.(ᵛ(x) - label), x.backprop)
    if y.backprop
        y.backward = function maeBackward()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* sign.(ᵛ(y))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


"""
    mse(x::Variable{T}, label::Variable{T}) -> y::Variable{T}

mean sqrt error (mse) between each element in the input `x` and target `label`. Also called L2Loss. i.e. ⤦\n
    y = (xᵢ - lᵢ)²
"""
function mse(x::Variable{T}, label::Variable{T}) where T
    @assert (x.shape == label.shape)
    backprop = (x.backprop || label.backprop)
    𝟚 = eltype(x)(2.0)
    y = Variable{T}((ᵛ(x) - ᵛ(label)).^𝟚, backprop)
    if backprop
        y.backward = function mseBackward()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* 𝟚 .* (ᵛ(x) - ᵛ(label))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function mse(x::Variable{T}, label::AbstractArray) where T
    @assert x.shape == size(label)
    𝟚 = eltype(x)(2.0)
    y = Variable{T}((ᵛ(x) - label).^𝟚, x.backprop)
    if y.backprop
        y.backward = function mseBackward()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* 𝟚 .* (ᵛ(x) - label)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


"""
    Lp(x::Variable{T}, label::Variable{T}; p=3) -> y::Variable{T}

absolute error's `p`-th power between each element in the input `x` and target `label`. Also called LpLoss. i.e. ⤦\n
    y = |xᵢ - lᵢ|ᵖ
"""
function Lp(x::Variable{T}, label::Variable{T}; p=3) where T
    @assert (x.shape == label.shape)
    backprop = (x.backprop || label.backprop)
    Δ = ᵛ(x) - ᵛ(label)
    y = Variable{T}(Δ .^ p, backprop)
    if backprop
        y.backward = function LpBackward()
            if need2computeδ!(x)
                i = (Δ .!= eltype(T)(0.0))
                x.delta[i] .+= y.delta[i] .* y.value[i] ./ Δ[i] .* p
                # δ(x) .+= δ(y) .* ᵛ(y) ./ Δ .* p
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function Lp(x::Variable{T}, label::AbstractArray; p=3) where T
    @assert x.shape == size(label)
    Δ = ᵛ(x) - label
    y = Variable{T}(Δ .^ p, x.backprop)
    if y.backprop
        y.backward = function LpBackward()
            if need2computeδ!(x)
                i = (Δ .!= eltype(T)(0))
                x.delta[i] .+= y.delta[i] .* y.value[i] ./ Δ[i] .* p
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


maeLoss(x::Variable{T}, label::Variable{T}; reduction::String="sum") where T = loss( mae(x, label), reduction=reduction )
maeLoss(x::Variable{T}, label::AbstractArray; reduction::String="sum") where T = loss( mae(x, label), reduction=reduction )
L1Loss(x::Variable{T},  label::Variable{T}; reduction::String="sum") where T = loss( mae(x, label), reduction=reduction )
L1Loss(x::Variable{T},  label::AbstractArray; reduction::String="sum") where T = loss( mae(x, label), reduction=reduction )

mseLoss(x::Variable{T}, label::Variable{T}; reduction::String="sum") where T = loss( mse(x, label), reduction=reduction )
mseLoss(x::Variable{T}, label::AbstractArray; reduction::String="sum") where T = loss( mse(x, label), reduction=reduction )
L2Loss(x::Variable{T},  label::Variable{T}; reduction::String="sum") where T = loss( mse(x, label), reduction=reduction )
L2Loss(x::Variable{T},  label::AbstractArray; reduction::String="sum") where T = loss( mse(x, label), reduction=reduction )

LpLoss(x::Variable{T}, label::Variable{T}; p=3, reduction::String="sum") where T = loss( Lp(x, label; p=p), reduction=reduction )
LpLoss(x::Variable{T}, label::AbstractArray; p=3, reduction::String="sum") where T = loss( Lp(x, label; p=p), reduction=reduction )
