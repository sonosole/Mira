## regression loss

export MAE
export MAELoss
export L1Loss

export MSE
export MSELoss
export L2Loss

export Lp, LpLoss


"""
    MAE(x::Variable{T}, label::Variable{T}) -> y::Variable{T}

mean absolute error (MAE) between each element in the input `x` and target `label`. Also called L1Loss. i.e. ⤦\n
    y = |xᵢ - lᵢ|
"""
function MAE(x::Variable{T}, label::Variable{T}) where T
    @assert (x.shape == label.shape)
    backprop = (x.backprop || label.backprop)
    y = Variable{T}(abs.(ᵛ(x) - ᵛ(label)), backprop)
    if backprop
        y.backward = function maeBackward()
            if need2computeδ!(x)
                x ← δ(y) .* sign.(ᵛ(y))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


"""
    MAE(x::Variable{T}, label::AbstractArray) -> y::Variable{T}

mean absolute error (MAE) between each element in the input `x` and target `label`. Also called L1Loss. i.e. ⤦\n
    y = |xᵢ - lᵢ|
"""
function MAE(x::Variable{T}, label::AbstractArray) where T
    @assert x.shape == size(label)
    y = Variable{T}(abs.(ᵛ(x) - label), x.backprop)
    if y.backprop
        y.backward = function maeBackward()
            if need2computeδ!(x)
                x ← δ(y) .* sign.(ᵛ(y))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


"""
    MAE(x::AbstractArray, label::AbstractArray) -> y::AbstractArray

mean absolute error (MAE) between each element in the input `x` and target `label`. Also called L1Loss. i.e. ⤦\n
    y = |xᵢ - lᵢ|
"""
function MAE(x::AbstractArray, label::AbstractArray)
    @assert size(x) == size(label)
    return abs.(x - label)
end


"""
    MSE(x::Variable{T}, label::Variable{T}) -> y::Variable{T}

mean sqrt error (MSE) between each element in the input `x` and target `label`. Also called L2Loss. i.e. ⤦\n
    y = (xᵢ - lᵢ)²
"""
function MSE(x::Variable{T}, label::Variable{T}) where T
    @assert (x.shape == label.shape)
    backprop = (x.backprop || label.backprop)
    𝟐 = eltype(x)(2)
    y = Variable{T}((ᵛ(x) - ᵛ(label)) .^ 𝟐, backprop)
    if backprop
        y.backward = function mseBackward()
            if need2computeδ!(x)
                x ← δ(y) .* 𝟐 .* (ᵛ(x) - ᵛ(label))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function MSE(x::Variable{T}, label::AbstractArray) where T
    @assert x.shape == size(label)
    𝟐 = eltype(x)(2.0)
    y = Variable{T}((ᵛ(x) - label) .^ 𝟐, x.backprop)
    if y.backprop
        y.backward = function mseBackward()
            if need2computeδ!(x)
                x ← δ(y) .* 𝟐 .* (ᵛ(x) - label)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function MSE(x::AbstractArray, label::AbstractArray)
    @assert size(x) == size(label)
    𝟐 = eltype(x)(2.0)
    return (x - label) .^ 𝟐
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
                # i = (Δ .!= eltype(T)(0.0))
                # x.delta[i] .+= y.delta[i] .* y.value[i] ./ Δ[i] .* p
                x ← δ(y) .* ᵛ(y) ./ Δ .* p
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
                # i = (Δ .!= eltype(T)(0))
                # x.delta[i] .+= y.delta[i] .* y.value[i] ./ Δ[i] .* p
                x ← δ(y) .* ᵛ(y) ./ Δ .* p
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function Lp(x::AbstractArray, label::AbstractArray; p=3)
    @assert size(x) == size(label)
    return (x - label) .^ p
end


MAELoss(x::Variable{T}, label::Variable{T}; reduction::String="sum") where T = Loss( MAE(x, label), reduction=reduction )
MAELoss(x::Variable{T}, label::AbstractArray; reduction::String="sum") where T = Loss( MAE(x, label), reduction=reduction )
MAELoss(x::AbstractArray, label::AbstractArray; reduction::String="sum") = Loss( MAE(x, label), reduction=reduction )

L1Loss(x::Variable{T},  label::Variable{T}; reduction::String="sum") where T = Loss( MAE(x, label), reduction=reduction )
L1Loss(x::Variable{T},  label::AbstractArray; reduction::String="sum") where T = Loss( MAE(x, label), reduction=reduction )
L1Loss(x::AbstractArray,  label::AbstractArray; reduction::String="sum") = Loss( MAE(x, label), reduction=reduction )

MSELoss(x::Variable{T}, label::Variable{T}; reduction::String="sum") where T = Loss( MSE(x, label), reduction=reduction )
MSELoss(x::Variable{T}, label::AbstractArray; reduction::String="sum") where T = Loss( MSE(x, label), reduction=reduction )
MSELoss(x::AbstractArray, label::AbstractArray; reduction::String="sum") = Loss( MSE(x, label), reduction=reduction )

L2Loss(x::Variable{T},  label::Variable{T}; reduction::String="sum") where T = Loss( MSE(x, label), reduction=reduction )
L2Loss(x::Variable{T},  label::AbstractArray; reduction::String="sum") where T = Loss( MSE(x, label), reduction=reduction )
L2Loss(x::AbstractArray,  label::AbstractArray; reduction::String="sum") = Loss( MSE(x, label), reduction=reduction )

LpLoss(x::Variable{T}, label::Variable{T}; p=3, reduction::String="sum") where T = Loss( Lp(x, label; p=p), reduction=reduction )
LpLoss(x::Variable{T}, label::AbstractArray; p=3, reduction::String="sum") where T = Loss( Lp(x, label; p=p), reduction=reduction )
LpLoss(x::AbstractArray, label::AbstractArray; p=3, reduction::String="sum") = Loss( Lp(x, label; p=p), reduction=reduction )
