export min2max, min2max!
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
    y = Variable{S}(min2max(ᵛ(x), lower=lower, upper=upper), x.backprop)
    if y.backprop
        y.backward = function ∇min2max()
            if needgrad(x)
                T = eltype(S)
                L = T(lower)
                U = T(upper)
                x ← δ(y) .* (L .< ᵛ(x) .< U)
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
            if needgrad(x)
                T = eltype(S)
                L = T(lower)
                U = T(upper)
                x ← δ(y) .* (L .< ᵛ(x) .< U)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

Base.clamp(x::AbstractArray, lo::Real, hi::Real) = clamp.(x, lo, hi)
Base.clamp(x::Variable, lo::Real, hi::Real)  = min2max(x, lower=lo, upper=hi)
Base.clamp!(x::Variable, lo::Real, hi::Real) = min2max!(x, lower=lo, upper=hi)


export sigmoid, sigmoid!
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
            if needgrad(x)
                x ← δ(y) .* ᵛ(y) .* (l .- ᵛ(y))
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
            if needgrad(x)
                x ← δ(y) .* ᵛ(y) .* (l .- ᵛ(y))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


export swish, swish!, silu, silu!
export hardswish, hardswish!
export mish, mish!

function swish!(x::AbstractArray)
    l = eltype(x)(1)
    @. x = x / (l + exp(-x))
end

function swish(x::AbstractArray)
    l = eltype(x)(1)
    return  x ./ (l .+ exp.(-x))
end

function swish!(x::Variable)
    return dotMul(sigmoid(x), x)
end

function swish(x::Variable)
    return dotMul(sigmoid(x), x)
end

silu(x) = swish(x)
silu!(x) = swish!(x)

function hardswish(x::T, o::T, 𝟑::T, _3::T, inv6::T) where T <: AbstractFloat
    x ≥ 𝟑 && return x
    x >_3 && return x * (x + 𝟑) * inv6
    return o
end

function ∂hardswish(x::T, o::T, l::T, 𝟑::T, _3::T, inv2::T, inv3::T) where T <: AbstractFloat
    x > 𝟑 && return l
    x >_3 && return inv3 * x + inv2
    return o
end

"""
    hardswish(x) = if x > 3
        x
    elseif x > -3
        1/6 * x * (x+3)
    else
        0
    end
"""
function hardswish(x::AbstractArray, α::Real=1.0f0)
    T = eltype(x)
    return hardswish.(x, T(0), T(3), T(-3), T(0.16666666666666666))
end

function hardswish!(x::AbstractArray, α::Real=1.0f0)
    T  = eltype(x)
    x .= hardswish.(x, T(0), T(3), T(-3), T(0.16666666666666666))
    return x
end

function hardswish(x::Variable{T}, α::Real=1.0f0) where T
    y = Variable{T}(hardswish(ᵛ(x), α), x.backprop)
    if y.backprop
        y.backward = function ∇hardswish()
            if needgrad(x)
                D = eltype(x)
                inv3 = D(0.3333333333333333)
                inv2 = D(0.50000000000000f0)
                x ← δ(y) .* ∂hardswish.(ᵛ(x), D(0), D(1), D(3), D(-3), inv2, inv3)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function hardswish!(x::Variable{T}, α::Real=1.0f0) where T
    y = Variable{T}(hardswish(ᵛ(x), α), x.backprop)
    if y.backprop
        y.backward = function ∇hardswish()
            if needgrad(x)
                D = eltype(x)
                inv3 = D(0.3333333333333333)
                inv2 = D(0.50000000000000f0)
                x ← δ(y) .* ∂hardswish.(ᵛ(x), D(0), D(1), D(3), D(-3), inv2, inv3)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function mish(x::AbstractArray)
    return @. x * tanh(softplus(x))
end

function mish!(x::AbstractArray)
    y = softplus(x)
    z = tanh!(y)
    return dotMul(x, z)
end

function mish!(x::Variable)
    y = softplus(x)
    z = tanh!(y)
    return dotMul(x, z)
end

function mish(x::Variable)
    y = softplus(x)
    z = tanh(y)
    return dotMul(x, z)
end


export softmax, softmin
## -------------------------------------------------------- softmax
function softmax(x::AbstractArray; dims::IntOrDims{N}=1) where N
    y = exp.(x .- maximum(x, dims=dims))
    Σ = eltype(x)(1.0f0) ./ sum(y, dims=dims)
    return y .* Σ
end

function softmax(x::Variable{T}; dims::IntOrDims{N}=1) where {T,N}
    y = Variable{T}(softmax(ᵛ(x); dims=dims), x.backprop)
    if y.backprop
        y.backward = function ∇softmax()
            if needgrad(x)
                ẏy = δ(y) .* ᵛ(y)
                x ← ẏy .- ᵛ(y) .* sum(ẏy, dims=dims)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function softmin(x::Union{AbstractArray,Variable}; dims::IntOrDims{N}=1) where N
    return softmax(-x; dims)
end


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
            if needgrad(x)
                x ← δ(y) ./ (l .+ exp.( - ᵛ(x) ))
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
            if needgrad(x)
                x ← δ(y) ./ (l .+ exp.( - ᵛ(x) ))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


export exp!
export exp2!
export exp10!

function exp!(x::Variable{T}) where T
    y = Variable{T}(exp!(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇exp()
            if needgrad(x)
                x ← δ(y) .* ᵛ(y)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function Base.exp(x::Variable{T}) where T
    y = Variable{T}(exp(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇exp()
            if needgrad(x)
                x ← δ(y) .* ᵛ(y)
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

function Base.exp2(x::AbstractArray)
    return exp2.(x)
end

function exp2!(x::Variable{T}) where T
    # exp2 represents y = 2^x
    y = Variable{T}(exp2!(ᵛ(x)), x.backprop)
    if x.backprop
        𝟚 = eltype(x)(2.0f0)
        y.backward = function ∇exp2()
            if needgrad(x)
                x ← δ(y) .* log(𝟚) .* ᵛ(y)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function Base.exp2(x::Variable{T}) where T
    # EXP2 represents y = 2^x
    y = Variable{T}(exp2(ᵛ(x)), x.backprop)
    if x.backprop
        𝟚 = eltype(x)(2.0f0)
        y.backward = function ∇exp2()
            if needgrad(x)
                x ← δ(y) .* log(𝟚) .* ᵛ(y)
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

function Base.exp10(x::AbstractArray)
    return exp10.(x)
end

function exp10!(x::Variable{T}) where T
    # EXP10 represents y = 10^x
    y = Variable{T}(exp10!(ᵛ(x)), x.backprop)
    if x.backprop
        lO = eltype(x)(10.0f0)
        y.backward = function ∇exp10()
            if needgrad(x)
                x ← δ(y) .* log(lO) .* ᵛ(y)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function Base.exp10(x::Variable{T}) where T
    # EXP10 represents y = 10^x
    y = Variable{T}(exp10(ᵛ(x)), x.backprop)
    if x.backprop
        lO = eltype(x)(10.0f0)
        y.backward = function ∇exp10()
            if needgrad(x)
                x ← δ(y) .* log(lO) .* ᵛ(y)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


export abs!
export sqrt!

function abs!(x::AbstractArray)
    @. x = abs(x)
end

function Base.abs(x::AbstractArray)
    return abs.(x)
end

function abs!(x::Variable{T}) where T
    y = Variable{T}(abs(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇abs()
            if needgrad(x)
                x ← δ(y) .* sign.(ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function Base.abs(x::Variable{T}) where T
    y = Variable{T}(abs(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇abs()
            if needgrad(x)
                x ← δ(y) .* sign.(ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function sqrt!(x::Variable{T}) where T
    ᵛ(x) .= sqrt!(ᵛ(x))
    y = Variable{T}(ᵛ(x), x.backprop)
    if x.backprop
        S = eltype(x)
        𝟚 = S(2.0f0)
        y.backward = function ∇sqrt()
            if needgrad(x)
                x ← δ(y) ./ (𝟚 .* ᵛ(y))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function Base.sqrt(x::Variable{T}) where T
    y = Variable{T}(sqrt(ᵛ(x)), x.backprop)
    if y.backprop
        S = eltype(x)
        𝟚 = S(2.0f0)
        y.backward = function ∇sqrt()
            if needgrad(x)
                x ← δ(y) ./ (𝟚 .* ᵛ(y))
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
            if needgrad(x)
                x ← - δ(y) .* ᵛ(y) .* ᵛ(y);
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function Base.inv(x::Variable{T}) where T
    y = Variable{T}(inv(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇inv()
            if needgrad(x)
                x ← - δ(y) .* ᵛ(y) .* ᵛ(y)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function Base.:/(constant::Real, x::Variable{T}) where T
    c = eltype(ᵛ(x))(constant)
    y = Variable{T}(c .* inv(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇inv()
            if needgrad(x)
                x ← - δ(y) .* ᵛ(y) .* ᵛ(y) .* (1/c)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

export log!
export log2!
export log10!

function log!(x::Variable{T}) where T
    y = Variable{T}(log(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇log()
            if needgrad(x)
                x ← δ(y) ./ ᵛ(x)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function Base.log(x::Variable{T}) where T
    y = Variable{T}(log(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇log()
            if needgrad(x)
                x ← δ(y) ./ ᵛ(x)
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

function Base.log2(x::AbstractArray)
    return log2.(x)
end

function log2!(x::Variable{T}) where T
    y = Variable{T}(log2(ᵛ(x)), x.backprop)
    if x.backprop
        𝟚 = eltype(x)(2.0f0)
        y.backward = function ∇log2()
            if needgrad(x)
                x ← δ(y) ./ (log(𝟚) .* ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function Base.log2(x::Variable{T}) where T
    y = Variable{T}(log2(ᵛ(x)), x.backprop)
    if x.backprop
        𝟚 = eltype(x)(2.0f0)
        y.backward = function ∇log2()
            if needgrad(x)
                x ← δ(y) ./ (log(𝟚) .* ᵛ(x))
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

function Base.log10(x::AbstractArray)
    return log10.(x)
end

function log10!(x::Variable{T}) where T
    y = Variable{T}(log10(ᵛ(x)), x.backprop)
    if x.backprop
        lO = eltype(x)(10.0f0)
        y.backward = function ∇log10()
            if needgrad(x)
                x ← δ(y) ./ (log(lO) .* ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function Base.log10(x::Variable{T}) where T
    y = Variable{T}(log10(ᵛ(x)), x.backprop)
    if x.backprop
        lO = eltype(x)(10.0f0)
        y.backward = function ∇log10()
            if needgrad(x)
                x ← δ(y) ./ (log(lO) .* ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

export sec!
function sec!(x::Variable{T}) where T
    # SEC represents y = sec(x)
    y = Variable{T}(sec(ᵛ(x)), x.backprop)
    if x.backprop
        y.backward = function ∇sec()
            if needgrad(x)
                x ← δ(y) .* ᵛ(y) .* tan.(ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function Base.sec(x::Variable{T}) where T
    # SEC represents y = sec(x)
    y = Variable{T}(sec(ᵛ(x)), x.backprop)
    if x.backprop
        y.backward = function ∇sec()
            if needgrad(x)
                x ← δ(y) .* ᵛ(y) .* tan.(ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


## -- tan serials --
export tan!
export atan!
export tand!
export tanh!
export tanhshrink, tanhshrink!
export hardtanh, hardtanh!

function tan!(x::Variable{T}) where T
    y = Variable{T}(tan!(ᵛ(x)), x.backprop)
    if y.backprop
        S = eltype(x)
        𝟙 = S(1.0)
        𝟚 = S(2.0)
        y.backward = function ∇tan()
            if needgrad(x)
                x ← δ(y) .* (𝟙 .+ ᵛ(y) .^ 𝟚)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function Base.tan(x::Variable{T}) where T
    y = Variable{T}(tan(ᵛ(x)), x.backprop)
    if y.backprop
        S = eltype(x)
        𝟙 = S(1.0)
        𝟚 = S(2.0)
        y.backward = function ∇tan()
            if needgrad(x)
                x ← δ(y) .* (𝟙 .+ ᵛ(y).^𝟚)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function atan!(x::Variable{T}) where T
    y = Variable{T}(atan.(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇atan()
            if needgrad(x)
                x ← δ(y) ./ (1 .+ ᵛ(x) .^ 2)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function Base.atan(x::Variable{T}) where T
    y = Variable{T}(atan.(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇atan()
            if needgrad(x)
                x ← δ(y) ./ (1 .+ ᵛ(x) .^ 2)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

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
    y = Variable{T}(hardtanh(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇hardtanh()
            if needgrad(x)
                x ← δ(y) .* (abs(ᵛ(x)) .< 1.0f0)
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
            if needgrad(x)
                x ← δ(y) .* (abs(ᵛ(x)) .< 1.0f0)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function tand!(x::AbstractArray)
    @. x = tand(x)
end

function Base.tand(x::AbstractArray)
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
            if needgrad(x)
                x ← δ(y) .* DPI .* (𝟙 .+ ᵛ(y).^𝟚)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function Base.tand(x::Variable{T}) where T
    y = Variable{T}(tand(ᵛ(x)), x.backprop)
    if y.backprop
        TOO = eltype(x)
        DPI = TOO(pi/180)
        𝟙 = TOO(1.0)
        𝟚 = TOO(2.0)
        y.backward = function ∇tand()
            if needgrad(x)
                x ← δ(y) .* DPI .* (𝟙 .+ ᵛ(y).^𝟚)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function tanh!(x::Variable{T}) where T
    y = Variable{T}(tanh!(ᵛ(x)), x.backprop)
    if y.backprop
        S = eltype(x)
        𝟙 = S(1.0f0)
        𝟚 = S(2.0f0)
        y.backward = function ∇tanh()
            if needgrad(x)
                x ← δ(y) .* (𝟙 .- ᵛ(y).^𝟚)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function Base.tanh(x::Variable{T}) where T
    y = Variable{T}(tanh(ᵛ(x)), x.backprop)
    if y.backprop
        S = eltype(x)
        𝟙 = S(1.0f0)
        𝟚 = S(2.0f0)
        y.backward = function ∇tanh()
            if needgrad(x)
                x ← δ(y) .* (𝟙 .- ᵛ(y).^𝟚)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

##
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
export asin!
export asinh!
export sinc!
export sind!
export sinh!
export sinpi!
export linearsin,linearsin!

function sin!(x::Variable{T}) where T
    y = Variable{T}(sin(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇sin()
            if needgrad(x)
                x ← δ(y) .* cos.(ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function Base.sin(x::Variable{T}) where T
    y = Variable{T}(sin(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇sin()
            if needgrad(x)
                x ← δ(y) .* cos.(ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function asin!(x::Variable{T}) where T
    y = Variable{T}(asin.(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇asin()
            if needgrad(x)
                x ← δ(y) ./ sqrt.(1 .- ᵛ(x) .^ 2)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function Base.asin(x::Variable{T}) where T
    y = Variable{T}(asin.(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇asin()
            if needgrad(x)
                x ← δ(y) ./ sqrt.(1 .- ᵛ(x) .^ 2)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function asinh!(x::Variable{T}) where T
    y = Variable{T}(asinh.(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇asinh()
            if needgrad(x)
                x ← δ(y) ./ sqrt.(ᵛ(x) .^ 2 .+ 1)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function Base.asinh(x::Variable{T}) where T
    y = Variable{T}(asinh.(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇asinh()
            if needgrad(x)
                x ← δ(y) ./ sqrt.(ᵛ(x) .^ 2 .+ 1)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end
##
function sinc!(x::AbstractArray)
    @. x = sinc(x)
end

function Base.sinc(x::AbstractArray)
    return sinc.(x)
end

function sinc!(x::Variable{T}) where T
    # sinc represents y = sin(pi*x)/(pi*x)
    y = Variable{T}(sinc(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇sinc()
            if needgrad(x)
                x ← δ(y) .* cosc.(ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function Base.sinc(x::Variable{T}) where T
    # sinc represents y = sin(pi*x)/(pi*x)
    y = Variable{T}(sinc(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇sinc()
            if needgrad(x)
                x ← δ(y) .* cosc.(ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

##
function sind!(x::AbstractArray)
    @. x = sind(x)
end

function Base.sind(x::AbstractArray)
    return sind.(x)
end

function sind!(x::Variable{T}) where T
    y = Variable{T}(sind(ᵛ(x)), x.backprop)
    if x.backprop
        DPI = eltype(x)(pi/180)
        y.backward = function ∇sind()
            if needgrad(x)
                x ← δ(y) .* DPI .* cosd.(ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function Base.sind(x::Variable{T}) where T
    y = Variable{T}(sind(ᵛ(x)), x.backprop)
    if y.backprop
        DPI = eltype(x)(pi/180) # 1 rad⁻¹
        y.backward = function ∇sind()
            if needgrad(x)
                x ← δ(y) .* DPI .* cosd.(ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

##
function sinpi!(x::AbstractArray)
    @. x = sinpi(x)
end

function Base.sinpi(x::AbstractArray)
    return sinpi.(x)
end

function sinpi!(x::Variable{T}) where T
    y = Variable{T}(sinpi(ᵛ(x)), x.backprop)
    if y.backprop
        𝝅 = eltype(x)(pi)
        y.backward = function ∇sinpi()
            if needgrad(x)
                x ← δ(y) .* 𝝅 .* cospi.(ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function Base.sinpi(x::Variable{T}) where T
    y = Variable{T}(sinpi(ᵛ(x)), x.backprop)
    if y.backprop
        𝝅 = eltype(x)(pi)
        y.backward = function ∇sinpi()
            if needgrad(x)
                x ← δ(y) .* 𝝅 .* cospi.(ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function sinh!(x::Variable{T}) where T
    y = Variable{T}(sinh.(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇sinh()
            if needgrad(x)
                x ← δ(y) .* cosh.(ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function Base.sinh(x::Variable{T}) where T
    y = Variable{T}(sinh.(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇sinh()
            if needgrad(x)
                x ← δ(y) .* cosh.(ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

##
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

##
export cos!
export cosh!
export acosh!
export acos!

function cos!(x::Variable{T}) where T
    y = Variable{T}(cos(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇cos()
            if needgrad(x)
                x ← - δ(y) .* sin.(ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function Base.cos(x::Variable{T}) where T
    y = Variable{T}(cos(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇cos()
            if needgrad(x)
                x ← - δ(y) .* sin.(ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function acos!(x::Variable{T}) where T
    y = Variable{T}(acos.(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇acos()
            if needgrad(x)
                x ← - δ(y) ./ sqrt.(1 .- ᵛ(x) .^ 2)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function Base.acos(x::Variable{T}) where T
    y = Variable{T}(acos.(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇acos()
            if needgrad(x)
                x ← - δ(y) ./ sqrt.(1 .- ᵛ(x) .^ 2)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function cosh!(x::Variable{T}) where T
    y = Variable{T}(cosh.(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇cosh()
            if needgrad(x)
                x ← δ(y) .* sinh.(ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function Base.cosh(x::Variable{T}) where T
    y = Variable{T}(cosh.(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇cosh()
            if needgrad(x)
                x ← δ(y) .* sinh.(ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function acosh!(x::Variable{T}) where T
    y = Variable{T}(acosh.(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇acosh()
            if needgrad(x)
                x ← δ(y) ./ sqrt.(ᵛ(x) .^ 2 .- 1)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function Base.acosh(x::Variable{T}) where T
    y = Variable{T}(acosh.(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇acosh()
            if needgrad(x)
                x ← δ(y) ./ sqrt.(ᵛ(x) .^ 2 .- 1)
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
            if needgrad(x)
                ẏy = δ(y) .* ᵛ(y)
                x ← (ẏy .- ᵛ(y) .* sum(ẏy, dims=dims)) .* k ./ ᵛ(x)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

export same
same(x) = x


##
export relu, relu!
export leakyrelu, leakyrelu!
export relu1, relu1!
export relu6, relu6!
export elu, elu!
export selu, selu!
export gelu, gelu!
export celu, celu!

function relu!(x::AbstractArray)
    @. x = max(0.0f0, x)
end

function relu(x::AbstractArray)
    O = eltype(x)(0.0f0)
    return max.(O, x)
end

function relu!(x::Variable{T}) where T
    y = Variable{T}(relu(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇relu()
            if needgrad(x)
                x ← δ(y) .* (ᵛ(x) .> 0.0f0)
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
            if needgrad(x)
                x ← δ(y) .* (ᵛ(x) .> 0.0f0)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function leakyrelu(x::T, k::T) where T <: Real
    x > 0 && return x
    return k * x
end

function ∂leakyrelu(x::T, k::T, o::T, l::T) where T <: Real
    x > o && return l
    return k
end

function leakyrelu!(x::AbstractArray, k::Real=0.01f0)
    k = eltype(x)(k)
    @. x = leakyrelu(x, k)
end

function leakyrelu(x::AbstractArray, k::Real=0.01f0)
    k = eltype(x)(k)
    return leakyrelu.(x, k)
end

function leakyrelu!(x::Variable{T}, k::Real=0.01f0) where T
    y = Variable{T}(leakyrelu(ᵛ(x), k), x.backprop)
    if y.backprop
        y.backward = function ∇leakyrelu()
            if needgrad(x)
                S = eltype(x)
                x ← δ(y) .* ∂leakyrelu.(ᵛ(x), S(k), S(0), S(1))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function leakyrelu(x::Variable{T}, k::Real=0.01f0) where T
    y = Variable{T}(leakyrelu(ᵛ(x), k), x.backprop)
    if y.backprop
        y.backward = function ∇leakyrelu()
            if needgrad(x)
                S = eltype(x)
                x ← δ(y) .* ∂leakyrelu.(ᵛ(x), S(k), S(0), S(1))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


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
    y = Variable{T}(relu1(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇relu1()
            if needgrad(x)
                x ← δ(y) .* (0.0f0 .< ᵛ(x) .< 1.0f0)
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
            if needgrad(x)
                x ← δ(y) .* (0.0f0 .< ᵛ(x) .< 1.0f0)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


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
    y = Variable{T}(relu6(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇relu6()
            if needgrad(x)
                x ← δ(y) .* (0.0f0 .< ᵛ(x) .< 6.0f0)
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
            if needgrad(x)
                x ← δ(y) .* (0.0f0 .< ᵛ(x) .< 6.0f0)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function elu(x::T, α::T) where T <: AbstractFloat
    x > 0 && return x
    return α * exp(x) - α
end

function ∂elu(x::T, y::T, α::T, o::T, l::T) where T <: AbstractFloat
    x > o && return l
    return y + α
end

function elu(x::AbstractArray, α::Real=1.0f0)
    T = eltype(x)
    return elu.(x, T(α))
end

function elu!(x::AbstractArray, α::Real=1.0f0)
    T  = eltype(x)
    x .= elu.(x, T(α))
    return x
end

function elu(x::Variable{T}, α::Real=1.0f0) where T
    y = Variable{T}(elu(ᵛ(x), α), x.backprop)
    if y.backprop
        y.backward = function ∇elu()
            if needgrad(x)
                S = eltype(x)
                x ← δ(y) .* ∂elu.(ᵛ(x), ᵛ(y), S(α), S(0), S(1))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function elu!(x::Variable{T}, α::Real=1.0f0) where T
    y = Variable{T}(elu(ᵛ(x), α), x.backprop)
    if y.backprop
        y.backward = function ∇elu()
            if needgrad(x)
                S = eltype(x)
                x ← δ(y) .* ∂elu.(ᵛ(x), ᵛ(y), S(α), S(0), S(1))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function selu(x::T, λ::T, α::T) where T <: AbstractFloat
    x > 0 && return λ * x
    λα = λ * α
    return λα * exp(x) - λα
end

function ∂selu(x::T, y::T, λ::T, α::T, o::T) where T <: AbstractFloat
    x > o && return λ
    return y + λ * α
end

"""
    selu(x) = if x > 0
        λ * x
    else
        λ * (α * eˣ - α)
    end
α = 1.6732632423543772848170429916717
λ = 1.0507009873554804934193349852946
"""
function selu(x::AbstractArray)
    T = eltype(x)
    λ = T(1.0507009873554804934193349852946)
    α = T(1.6732632423543772848170429916717)
    return selu.(x, λ, α)
end

function selu!(x::AbstractArray)
    T  = eltype(x)
    λ  = T(1.0507009873554804934193349852946)
    α  = T(1.6732632423543772848170429916717)
    x .= selu.(x, λ, α)
    return x
end

function selu(x::Variable{T}) where T
    y = Variable{T}(selu(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇selu()
            if needgrad(x)
                S = eltype(x)
                λ = S(1.0507009873554804934193349852946)
                α = S(1.6732632423543772848170429916717)
                x ← δ(y) .* ∂selu.(ᵛ(x), ᵛ(y), λ, α, S(0))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function selu!(x::Variable{T}) where T
    y = Variable{T}(selu(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇selu()
            if needgrad(x)
                S = eltype(x)
                λ = S(1.0507009873554804934193349852946)
                α = S(1.6732632423543772848170429916717)
                x ← δ(y) .* ∂selu.(ᵛ(x), ᵛ(y), λ, α, S(0))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


"""
y = 0.5 ∗ x ∗ (1 + tanh( sqrt(2/π) * (x + 0.044715 * x³)) )\n
     a                         b              c
"""
function gelu(x::AbstractArray)
    T = eltype(x)
    l = T(1)
    a = T(0.5)
    b = T(0.7978845608028654)
    c = T(0.044715)
    return @. a * x * (l + tanh( b * (x + c * x^3)) )
end

function gelu!(x::AbstractArray)
    T = eltype(x)
    l = T(1)
    a = T(0.5)
    b = T(0.7978845608028654)
    c = T(0.044715)
    @. x = a * x * (l + tanh( b * (x + c * x^3)) )
    return x
end

"""
t = 0.0356774*x³ + 0.797885*x
            a             b
y = 0.5 * tanh(t) + (0.053516*x³ + 0.398942*x)*sech²(t) + 0.5
      c                     d             e                 c
"""
function ∂gelu(x::AbstractArray)
    T = eltype(x)
    a = T(0.0356774)
    b = T(0.7978850)
    c = T(0.5000000)
    d = T(0.0535160)
    e = T(0.3989420)
    t = @. a*x^3 + b*x
    y = @. c * tanh(t) + (d*x^3 + e*x)*sech(t)^2 + c
    return y
end

"""
    gelu(x) = 0.5 ∗ x ∗ (1 + tanh( sqrt(2/π) * (x + 0.044715 * x³)) )
"""
function gelu(x::Variable{T}) where T
    y = Variable{T}(gelu(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇gelu()
            if needgrad(x)
                x ← δ(y) .* ∂gelu(ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function gelu!(x::Variable{T}) where T
    y = Variable{T}(gelu(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇gelu()
            if needgrad(x)
                x ← δ(y) .* ∂gelu(ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function celu(x::T, α::T, o::T, l::T) where T <: AbstractFloat
    x > o && return x
    return α * (exp(x/α) - l)
end

function ∂celu(x::T, y::T, α::T, o::T, l::T) where T <: AbstractFloat
    x > o && return l
    return y / α + l
end

"""
    celu(x, α) = if x > 0
        x
    else
        α(exp(x/α) - 1)
    end
"""
function celu(x::AbstractArray, α::Real=1.0f0)
    T = eltype(x)
    return celu.(x, T(α), T(0), T(1))
end

function celu!(x::AbstractArray, α::Real=1.0f0)
    T  = eltype(x)
    x .= celu.(x, T(α), T(0), T(1))
    return x
end

function celu(x::Variable{T}, α::Real=1.0f0) where T
    y = Variable{T}(celu(ᵛ(x), α), x.backprop)
    if y.backprop
        y.backward = function ∇celu()
            if needgrad(x)
                D = eltype(x)
                x ← δ(y) .* ∂celu.(ᵛ(x), ᵛ(y), D(α), D(0), D(1))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function celu!(x::Variable{T}, α::Real=1.0f0) where T
    y = Variable{T}(celu(ᵛ(x), α), x.backprop)
    if y.backprop
        y.backward = function ∇celu()
            if needgrad(x)
                D = eltype(x)
                x ← δ(y) .* ∂celu.(ᵛ(x), ᵛ(y), D(α), D(0), D(1))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end
