# ----------------------------------------
#      activation functions
# ----------------------------------------

export min2max, min2max!
## -------------------------------------------------------- min2max
"""
    min2max!(x::AbstractArray; lower=0.0, upper=1.0) -> x

limit the scope of the data, i.e. ⤦\n
    @. x = min(max(x, lower), upper)
"""
function min2max!(x::AbstractArray; lower::Real=0.0f0, upper::Real=1.0f0)
    T = eltype(x)
    L = T(lower)
    U = T(upper)
    @. x = min(max(x, L), U)
end


"""
    min2max(x::AbstractArray; lower=0.0, upper=1.0) -> y

limit the scope of the data, i.e. ⤦\n
    y = min.(max.(x, lower), upper)
"""
function min2max(x::AbstractArray; lower::Real=0.0f0, upper::Real=1.0f0)
    T = eltype(x)
    L = T(lower)
    U = T(upper)
    return min.(max.(x, L), U)
end


"""
    min2max!(x::Variable{S}; lower=0.0, upper=1.0) -> y::Variable{S}

limit the scope of the data, i.e. ⤦\n
    y = Variable{S}(min2max!(ᵛ(x), lower=lower, upper=upper), x.backprop)
"""
function min2max!(x::Variable{S}; lower::Real=0.0f0, upper::Real=1.0f0) where S
    y = Variable{S}(min2max!(ᵛ(x), lower=lower, upper=upper), x.backprop)
    if y.backprop
        y.backward = function ∇min2max()
            if need2computeδ!(x)
                T = eltype(S)
                L = T(lower)
                U = T(upper)
                δ(x) .+= δ(y) .* (L .< ᵛ(x) .< U)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


"""
    min2max(x::Variable{S}; lower=0.0, upper=1.0) where S -> y::Variable{S}

limit the scope of the data, i.e. ⤦\n
    y = Variable{S}(min2max(ᵛ(x), lower=lower, upper=upper), x.backprop)
"""
function min2max(x::Variable{S}; lower::Real=0.0f0, upper::Real=1.0f0) where S
    y = Variable{S}(min2max(ᵛ(x), lower=lower, upper=upper), x.backprop)
    if x.backprop
        y.backward = function ∇min2max()
            if need2computeδ!(x)
                T = eltype(S)
                L = T(lower)
                U = T(upper)
                δ(x) .+= δ(y) .* (L .< ᵛ(x) .< U)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


export relu, relu!
## -------------------------------------------------------- relu
function relu!(x::AbstractArray)
    @. x = max(0.0f0, x)
end


function relu(x::AbstractArray)
    O = eltype(x)(0.0f0)
    return max.(O, x)
end


function relu!(x::Variable{T}) where T
    y = Variable{T}(relu!(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇relu()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* (ᵛ(x) .> 0.0f0)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function relu(x::Variable{T}) where T
    y = Variable{T}(relu(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇relu()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* (ᵛ(x) .> 0.0f0)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


export relu1, relu1!
## -------------------------------------------------------- relu1
function relu1!(x::AbstractArray)
    @. x = min(max(x, 0.0f0), 1.0f0)
end


function relu1(x::AbstractArray)
    T = eltype(x)
    O = T(0.0f0)
    l = T(1.0f0)
    return min.(max.(x, O), l)
end


function relu1!(x::Variable{T}) where T
    y = Variable{T}(relu1!(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇relu1()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* (0.0f0 .< ᵛ(x) .< 1.0f0)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function relu1(x::Variable{T}) where T
    y = Variable{T}(relu1(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇relu1()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* (0.0f0 .< ᵛ(x) .< 1.0f0)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


export relu6, relu6!
## -------------------------------------------------------- relu6
function relu6!(x::AbstractArray)
    @. x = min(max(x, 0.0f0), 6.0f0)
end


function relu6(x::AbstractArray)
    T = eltype(x)
    𝟘 = T(0.0f0)
    𝟞 = T(6.0f0)
    return min.(max.(x, 𝟘), 𝟞)
end


function relu6!(x::Variable{T}) where T
    y = Variable{T}(relu6!(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇relu6()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* (0.0f0 .< ᵛ(x) .< 6.0f0)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function relu6(x::Variable{T}) where T
    y = Variable{T}(relu6(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇relu6()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* (0.0f0 .< ᵛ(x) .< 6.0f0)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


export hardtanh, hardtanh!
## -------------------------------------------------------- hardtanh
function hardtanh!(x::AbstractArray)
    T  = eltype(x)
    l₋ = T(-1.0)
    l₊ = T( 1.0)
    @. x = min(max(x, l₋), l₊)
end


function hardtanh(x::AbstractArray)
    T = eltype(x)
    l₋ = T(-1.0)
    l₊ = T( 1.0)
    return min.(max.(x, l₋), l₊)
end


function hardtanh!(x::Variable{T}) where T
    y = Variable{T}(hardtanh!(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇hardtanh()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* (abs(ᵛ(x)) .< 1.0f0)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function hardtanh(x::Variable{T}) where T
    y = Variable{T}(hardtanh(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇hardtanh()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* (abs(ᵛ(x)) .< 1.0f0)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


export leakyrelu, leakyrelu!
## -------------------------------------------------------- leakyrelu
function leakyrelu!(x::AbstractArray)
    ZPONE = eltype(x)(0.1f0)
    @. x = max(ZPONE * x, x)
end


function leakyrelu(x::AbstractArray)
    ZPONE = eltype(x)(0.1f0)
    return max.(ZPONE .* x, x)
end


function leakyrelu!(x::Variable{T}) where T
    ZPONE = eltype(x)(0.1f0)
    tempv = ᵛ(x) .* ZPONE
    ᵛ(x) .= max.(ᵛ(x), tempv)
    y = Variable{T}(ᵛ(x), x.backprop)
    if y.backprop
        mask1 = ᵛ(x) .> tempv
        mask2 = .!mask1
        y.backward = function ∇leakyrelu()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* (mask1 .+ ZPONE .* mask2)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function leakyrelu(x::Variable{T}) where T
    ZPONE = eltype(x)(0.1f0)
    tempv = ᵛ(x) .* ZPONE
    mask1 = ᵛ(x) .> tempv
    mask2 = .!mask1
    y = Variable{T}(max.(tempv, ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇leakyrelu()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* (mask1 + ZPONE .* mask2)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


export sigmoid, sigmoid!
## -------------------------------------------------------- sigmoid
function sigmoid!(x::AbstractArray)
    l = eltype(x)(1.0f0)
    @. x = l / (l + exp(-x))
end


function sigmoid(x::AbstractArray)
    l = eltype(x)(1.0f0)
    return l ./ (l .+ exp.(-x))
end


function sigmoid!(x::Variable{T}) where T
    y = Variable{T}(sigmoid!(ᵛ(x)), x.backprop)
    if y.backprop
        l = eltype(x)(1.0f0)
        y.backward = function ∇sigmoid()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* ᵛ(y) .* (l .- ᵛ(y))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function sigmoid(x::Variable{T}) where T
    y = Variable{T}(sigmoid(ᵛ(x)), x.backprop)
    if y.backprop
        l = eltype(x)(1.0f0)
        y.backward = function ∇sigmoid()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* ᵛ(y) .* (l .- ᵛ(y))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


export swish, swish!
## -------------------------------------------------------- swish
function swish!(x::AbstractArray)
    l = eltype(x)(1.0f0)
    @. x = x / (l + exp(-x))
end


function swish(x::AbstractArray)
    l = eltype(x)(1.0f0)
    return  x ./ (l .+ exp.(-x))
end


function swish!(x::Variable{T}) where T
    return dotMul(sigmoid(x), x)
end


function swish(x::Variable{T}) where T
    return dotMul(sigmoid(x), x)
end


export softmax
## -------------------------------------------------------- softmax
function softmax(x::AbstractArray; dims::Union{Int,NTuple{N,Int}}=1) where N
    y = exp.(x .- maximum(x, dims=dims))
    Σ = eltype(x)(1.0f0) ./ sum(y, dims=dims)
    return y .* Σ
end


function softmax(x::Variable{T}; dims::Union{Int,NTuple{N,Int}}=1) where {T,N}
    y = Variable{T}(softmax(ᵛ(x); dims=dims), x.backprop)
    if y.backprop
        y.backward = function ∇softmax()
            if need2computeδ!(x)
                ẏy = δ(y) .* ᵛ(y)
                δ(x) .+= ẏy .- ᵛ(y) .* sum(ẏy, dims=dims)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


# -----------------
# 不常用激活函数....
# -----------------
export softplus, softplus!
function softplus!(x::AbstractArray)
    @. x = log(1.0f0 + exp(x))
end


function softplus(x::AbstractArray)
    l = eltype(x)(1.0f0)
    return log.( l .+ exp.(x) )
end


function softplus!(x::Variable{T}) where T
    y = Variable{T}(softplus(ᵛ(x)), x.backprop)
    if y.backprop
        l = eltype(x)(1.0f0)
        y.backward = function ∇softplus()
            if need2computeδ!(x)
                δ(x) .+= δ(y) ./ (l .+ exp.( - ᵛ(x) ))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function softplus(x::Variable{T}) where T
    y = Variable{T}(softplus(ᵛ(x)), x.backprop)
    if y.backprop
        l = eltype(x)(1.0f0)
        y.backward = function ∇softplus()
            if need2computeδ!(x)
                δ(x) .+= δ(y) ./ (l .+ exp.( - ᵛ(x) ))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


export exp!
function exp!(x::Variable{T}) where T
    y = Variable{T}(exp!(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇exp()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* ᵛ(y)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function Base.:exp(x::Variable{T}) where T
    y = Variable{T}(exp(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇exp()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* ᵛ(y)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


export log!
function log!(x::Variable{T}) where T
    y = Variable{T}(log(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇log()
            if need2computeδ!(x)
                δ(x) .+= δ(y) ./ ᵛ(x)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function Base.:log(x::Variable{T}) where T
    y = Variable{T}(log(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇log()
            if need2computeδ!(x)
                δ(x) .+= δ(y) ./ ᵛ(x)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


export abs!
function abs!(x::AbstractArray)
    @. x = abs(x)
end


function Base.:abs(x::AbstractArray)
    return abs.(x)
end


function abs!(x::Variable{T}) where T
    y = Variable{T}(abs(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇abs()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* sign.(ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function Base.:abs(x::Variable{T}) where T
    y = Variable{T}(abs(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇abs()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* sign.(ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function Base.:reshape(x::Variable{T}, newsize) where T
    y = Variable{T}( reshape(ᵛ(x), newsize), x.backprop )
    if y.backprop
        y.backward = function ∇reshape()
            if need2computeδ!(x)
                δ(x) .+= reshape(δ(y), x.shape)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function exp2!(x::AbstractArray)
    @. x = exp2(x)
end


function Base.:exp2(x::AbstractArray)
    return exp2.(x)
end


function exp2!(x::Variable{T}) where T
    # exp2 represents y = 2^x
    y = Variable{T}(exp2!(ᵛ(x)), x.backprop)
    if x.backprop
        𝟚 = eltype(x)(2.0f0)
        y.backward = function ∇exp2()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* log(𝟚) .* ᵛ(y)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function Base.:exp2(x::Variable{T}) where T
    # EXP2 represents y = 2^x
    y = Variable{T}(exp2(ᵛ(x)), x.backprop)
    if x.backprop
        𝟚 = eltype(x)(2.0f0)
        y.backward = function ∇exp2()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* log(𝟚) .* ᵛ(y)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function exp10!(x::AbstractArray)
    @. x = exp10(x)
end


function Base.:exp10(x::AbstractArray)
    return exp10.(x)
end


function exp10!(x::Variable{T}) where T
    # EXP10 represents y = 10^x
    y = Variable{T}(exp10!(ᵛ(x)), x.backprop)
    if x.backprop
        lO = eltype(x)(10.0f0)
        y.backward = function ∇exp10()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* log(lO) .* ᵛ(y)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function Base.:exp10(x::Variable{T}) where T
    # EXP10 represents y = 10^x
    y = Variable{T}(exp10(ᵛ(x)), x.backprop)
    if x.backprop
        lO = eltype(x)(10.0f0)
        y.backward = function ∇exp10()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* log(lO) .* ᵛ(y)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function log2!(x::AbstractArray)
    @. x = log2(x)
end


function Base.:log2(x::AbstractArray)
    return log2.(x)
end


function log2!(x::Variable{T}) where T
    y = Variable{T}(log2(ᵛ(x)), x.backprop)
    if x.backprop
        𝟚 = eltype(x)(2.0f0)
        y.backward = function ∇log2()
            if need2computeδ!(x)
                δ(x) .+= δ(y) ./ (log(𝟚) .* ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function Base.:log2(x::Variable{T}) where T
    y = Variable{T}(log2(ᵛ(x)), x.backprop)
    if x.backprop
        𝟚 = eltype(x)(2.0f0)
        y.backward = function ∇log2()
            if need2computeδ!(x)
                δ(x) .+= δ(y) ./ (log(𝟚) .* ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function log10!(x::AbstractArray)
    @. x = log10(x)
end


function Base.:log10(x::AbstractArray)
    return log10.(x)
end


function log10!(x::Variable{T}) where T
    y = Variable{T}(log10(ᵛ(x)), x.backprop)
    if x.backprop
        lO = eltype(x)(10.0f0)
        y.backward = function ∇log10()
            if need2computeδ!(x)
                δ(x) .+= δ(y) ./ (log(lO) .* ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function Base.:log10(x::Variable{T}) where T
    y = Variable{T}(log10(ᵛ(x)), x.backprop)
    if x.backprop
        lO = eltype(x)(10.0f0)
        y.backward = function ∇log10()
            if need2computeδ!(x)
                δ(x) .+= δ(y) ./ (log(lO) .* ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function sec!(x::Variable{T}) where T
    # SEC represents y = sec(x)
    y = Variable{T}(sec(ᵛ(x)), x.backprop)
    if x.backprop
        y.backward = function ∇sec()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* ᵛ(y) .* tan.(ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function Base.:sec(x::Variable{T}) where T
    # SEC represents y = sec(x)
    y = Variable{T}(sec(ᵛ(x)), x.backprop)
    if x.backprop
        y.backward = function ∇sec()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* ᵛ(y) .* tan.(ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function sqrt!(x::Variable{T}) where T
    y = Variable{T}(sqrt!(ᵛ(x)), x.backprop)
    if x.backprop
        S = eltype(x)
        𝟚 = S(2.0f0)
        ϵ = S(1e-38)
        y.backward = function ∇sqrt()
            if need2computeδ!(x)
                δ(x) .+= δ(y) ./ (𝟚 .* (ᵛ(y) .+ ϵ))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function Base.:sqrt(x::Variable{T}) where T
    y = Variable{T}(sqrt(ᵛ(x)), x.backprop)
    if y.backprop
        S = eltype(x)
        𝟚 = S(2.0f0)
        ϵ = S(1e-38)
        y.backward = function ∇sqrt()
            if need2computeδ!(x)
                δ(x) .+= δ(y) ./ (𝟚 .* (ᵛ(y) .+ ϵ))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


# -- tan serials --
export tan!
function tan!(x::Variable{T}) where T
    y = Variable{T}(tan!(ᵛ(x)), x.backprop)
    if y.backprop
        S = eltype(x)
        𝟙 = S(1.0)
        𝟚 = S(2.0)
        y.backward = function ∇tan()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* (𝟙 .+ ᵛ(y).^𝟚)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function Base.:tan(x::Variable{T}) where T
    y = Variable{T}(tan(ᵛ(x)), x.backprop)
    if y.backprop
        S = eltype(x)
        𝟙 = S(1.0)
        𝟚 = S(2.0)
        y.backward = function ∇tan()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* (𝟙 .+ ᵛ(y).^𝟚)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


export tand!
function tand!(x::AbstractArray)
    @. x = tand(x)
end


function Base.:tand(x::AbstractArray)
    return tand.(x)
end


function tand!(x::Variable{T}) where T
    y = Variable{T}(tand!(ᵛ(x)), x.backprop)
    if y.backprop
        TOO = eltype(x)
        DPI = TOO(pi/180)
        𝟙 = TOO(1.0)
        𝟚 = TOO(2.0)
        y.backward = function ∇tand()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* DPI .* (𝟙 .+ ᵛ(y).^𝟚)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function Base.:tand(x::Variable{T}) where T
    y = Variable{T}(tand(ᵛ(x)), x.backprop)
    if y.backprop
        TOO = eltype(x)
        DPI = TOO(pi/180)
        𝟙 = TOO(1.0)
        𝟚 = TOO(2.0)
        y.backward = function ∇tand()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* DPI .* (𝟙 .+ ᵛ(y).^𝟚)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


export tanh!
function tanh!(x::Variable{T}) where T
    y = Variable{T}(tanh!(ᵛ(x)), x.backprop)
    if y.backprop
        S = eltype(x)
        𝟙 = S(1.0f0)
        𝟚 = S(2.0f0)
        y.backward = function ∇tanh()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* (𝟙 .- ᵛ(y).^𝟚)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function Base.:tanh(x::Variable{T}) where T
    y = Variable{T}(tanh(ᵛ(x)), x.backprop)
    if y.backprop
        S = eltype(x)
        𝟙 = S(1.0f0)
        𝟚 = S(2.0f0)
        y.backward = function ∇tanh()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* (𝟙 .- ᵛ(y).^𝟚)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


export tanhshrink, tanhshrink!
function tanhshrink!(x::AbstractArray)
    @. x = x - tanh(x)
end


function tanhshrink(x::AbstractArray)
    return  x - tanh(x)
end


function tanhshrink!(x::Variable{T}) where T
    return x - tanh(x)
end


function tanhshrink(x::Variable{T}) where T
    return x - tanh(x)
end


## -- sin serials --
export sin!
function sin!(x::Variable{T}) where T
    y = Variable{T}(sin(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇sin()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* cos.(ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function Base.:sin(x::Variable{T}) where T
    y = Variable{T}(sin(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇sin()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* cos.(ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


export sinc!
function sinc!(x::AbstractArray)
    @. x = sinc(x)
end


function Base.:sinc(x::AbstractArray)
    return sinc.(x)
end


function sinc!(x::Variable{T}) where T
    # sinc represents y = sin(pi*x)/(pi*x)
    y = Variable{T}(sinc(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇sinc()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* cosc.(ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function Base.:sinc(x::Variable{T}) where T
    # sinc represents y = sin(pi*x)/(pi*x)
    y = Variable{T}(sinc(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇sinc()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* cosc.(ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


export sind!
function sind!(x::AbstractArray)
    @. x = sind(x)
end


function Base.:sind(x::AbstractArray)
    return sind.(x)
end


function sind!(x::Variable{T}) where T
    y = Variable{T}(sind(ᵛ(x)), x.backprop)
    if x.backprop
        DPI = eltype(x)(pi/180)
        y.backward = function ∇sind()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* DPI .* cosd.(ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function Base.:sind(x::Variable{T}) where T
    y = Variable{T}(sind(ᵛ(x)), x.backprop)
    if y.backprop
        DPI = eltype(x)(pi/180) # 1 rad⁻¹
        y.backward = function ∇sind()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* DPI .* cosd.(ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


export sinpi!
function sinpi!(x::AbstractArray)
    @. x = sinpi(x)
end


function Base.:sinpi(x::AbstractArray)
    return sinpi.(x)
end


function sinpi!(x::Variable{T}) where T
    y = Variable{T}(sinpi(ᵛ(x)), x.backprop)
    if y.backprop
        𝝅 = eltype(x)(pi)
        y.backward = function ∇sinpi()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* 𝝅 .* cospi.(ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function Base.:sinpi(x::Variable{T}) where T
    y = Variable{T}(sinpi(ᵛ(x)), x.backprop)
    if y.backprop
        𝝅 = eltype(x)(pi)
        y.backward = function ∇sinpi()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* 𝝅 .* cospi.(ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


export linearsin,linearsin!
function linearsin!(x::AbstractArray)
    @. x = sin(x) + x
end


function linearsin(x::AbstractArray)
    return sin(x) + x
end


function linearsin!(x::Variable{T}) where T
    return sin(x) + x
end


function linearsin(x::Variable{T}) where T
    return sin(x) + x
end


export cos!
function cos!(x::Variable{T}) where T
    y = Variable{T}(cos(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇cos()
            if need2computeδ!(x)
                δ(x) .+= - δ(y) .* sin.(ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function Base.:cos(x::Variable{T}) where T
    y = Variable{T}(cos(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇cos()
            if need2computeδ!(x)
                δ(x) .+= - δ(y) .* sin.(ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


export inv!
function inv!(x::Variable{T}) where T
    y = Variable{T}(inv!(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇inv()
            if need2computeδ!(x)
                δ(x) .+= - δ(y) .* ᵛ(y) .* ᵛ(y);
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function Base.:inv(x::Variable{T}) where T
    y = Variable{T}(inv(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇inv()
            if need2computeδ!(x)
                δ(x) .+= - δ(y) .* ᵛ(y) .* ᵛ(y)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

export polymax
"""
    polymax(x::AbstractArray, n::Int; dims=1) -> y::AbstractArray
`y = xⁿ ./ sum(xⁿ, dims=dims)`
"""
function polymax(x::AbstractArray, n::Int; dims::Union{Int,NTuple{N,Int}}=1) where N
    xⁿ = x .^ n
    return xⁿ ./ sum(xⁿ, dims=dims)
end


"""
    polymax(x::Variable, n::Int; dims=1) -> y::Variable
`y = xⁿ ./ sum(xⁿ, dims=dims)`
"""
function polymax(x::Variable{T}, n::Int; dims::Union{Int,NTuple{N,Int}}=1) where {T,N}
    y = Variable{T}(polymax(ᵛ(x), n; dims=dims), x.backprop)
    if y.backprop
        S = eltype(ᵛ(x))
        k = S(n)
        y.backward = function ∇softmax()
            if need2computeδ!(x)
                ẏy = δ(y) .* ᵛ(y)
                δ(x) .+= (ẏy .- ᵛ(y) .* sum(ẏy, dims=dims)) .* k ./ ᵛ(x)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

export same
same(x) = x
