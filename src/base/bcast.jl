"""
    axes2reduce(z, x) -> axes::Vector{Int}
axes need to be reduced, `z` and `x` comes from `z = broadcast(::typeof(+-*/...), x, y)`\n

# Example
    axes2reduce(rand(3,4,5),rand(1,4))    -> (1,3)
    axes2reduce(rand(3,4,5),rand(1,4,1))  -> (1,3)
"""
function axes2reduce(z, x)
    a = Int[]
    for i = 1:ndims(z)
        if size(x, i) == 1
            push!(a, i)
        end
    end
    return a
end


"""
    unbcast(δx::AbstractArray, x::AbstractArray) -> ∇x

reduced `δx` to `∇x` according to shape difference from `x` and `δx`

# Params
`x`  : comes from `z = broadcast(::typeof(+-*/...), x, y)`\n
`δx` : unreduced gradient, i.e. `δx = δz .* ∂z/∂x`\n
`∇x` : reduced gradient, i.e. ⤓⤓⤓\n
       Δx = sum(δx, dims=axes2reduce(δx, x)) # reduced but still has redundant dimensions\n
       ∇x = reshape(Δx, size(x))
"""
function unbcast(δx::AbstractArray, x::AbstractArray)
    if size(δx) == size(x)
        return δx
    elseif length(δx) == length(x)
        return reshape(δx, size(x))
    else
        Δx = sum(δx, dims=axes2reduce(δx,x))
        return reshape(Δx, size(x))
    end
end

import Base.Broadcast.broadcasted

# z = x .+ y
function broadcasted(::typeof(+), x::Variable{T1}, y::Variable{T2}) where {T1,T2}
    @assert T1 <: T2 || T1 >: T2
    T = T1 <: T2 ? T1 : T2
    z = Variable{T}(ᵛ(x) .+ ᵛ(y), x.backprop || y.backprop)
    if z.backprop
        z.backward = function DotAddBackward()
            δz = copy(δ(z))
            if need2computeδ!(x)
                δx = δz
                δ(x) .+= unbcast(δx, ᵛ(x))
            end
            if need2computeδ!(y)
                δy = δz
                δ(y) .+= unbcast(δy, ᵛ(y))
            end
            ifNotKeepδThenFreeδ!(z)
        end
        addchild(z, x)
        addchild(z, y)
    end
    return z
end


function broadcasted(::typeof(+), x::Variable{T1}, y::T2) where {T1,T2}
    @assert T1 <: T2 || T1 >: T2
    T = T1 <: T2 ? T1 : T2
    z = Variable{T}(ᵛ(x) .+ y, x.backprop)
    if z.backprop
        z.backward = function DotAddBackward()
            if need2computeδ!(x)
                δx = copy(δ(z))
                δ(x) .+= unbcast(δx, ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(z)
        end
        addchild(z, x)
    end
    return z
end


function broadcasted(::typeof(+), x::T1, y::Variable{T2}) where {T1,T2}
    @assert T1 <: T2 || T1 >: T2
    T = T1 <: T2 ? T1 : T2
    z = Variable{T}(x .+ ᵛ(y), y.backprop)
    if z.backprop
        z.backward = function DotAddBackward()
            if need2computeδ!(y)
                δy = copy(δ(z))
                δ(y) .+= unbcast(δy, ᵛ(y))
            end
            ifNotKeepδThenFreeδ!(z)
        end
        addchild(z, y)
    end
    return z
end


# z = x .- y
function broadcasted(::typeof(-), x::Variable{T1}, y::Variable{T2}) where {T1,T2}
    @assert T1 <: T2 || T1 >: T2
    T = T1 <: T2 ? T1 : T2
    z = Variable{T}(ᵛ(x) .- ᵛ(y), x.backprop || y.backprop)
    if z.backprop
        z.backward = function DotMinusBackward()
            if need2computeδ!(x)
                δx = copy(δ(z))
                δ(x) .+= unbcast(δx, ᵛ(x))
            end
            if need2computeδ!(y)
                δy = - δ(z)
                δ(y) .+= unbcast(δy, ᵛ(y))
            end
            ifNotKeepδThenFreeδ!(z)
        end
        addchild(z, x)
        addchild(z, y)
    end
    return z
end


function broadcasted(::typeof(-), x::Variable{T1}, y::T2) where {T1,T2}
    @assert T1 <: T2 || T1 >: T2
    T = T1 <: T2 ? T1 : T2
    z = Variable{T}(ᵛ(x) .- y, x.backprop)
    if z.backprop
        z.backward = function DotMinusBackward()
            if need2computeδ!(x)
                δx = copy(δ(z))
                δ(x) .+= unbcast(δx, ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(z)
        end
        addchild(z, x)
    end
    return z
end


function broadcasted(::typeof(-), x::T1, y::Variable{T2}) where {T1,T2}
    @assert T1 <: T2 || T1 >: T2
    T = T1 <: T2 ? T1 : T2
    z = Variable{T}(x .- ᵛ(y), y.backprop)
    if z.backprop
        z.backward = function DotMinusBackward()
            if need2computeδ!(y)
                δy = - δ(z)
                δ(y) .+= unbcast(δy, ᵛ(y))
            end
            ifNotKeepδThenFreeδ!(z)
        end
        addchild(z, y)
    end
    return z
end


# z = x .* y
function broadcasted(::typeof(*), x::Variable{T1}, y::Variable{T2}) where {T1,T2}
    @assert T1 <: T2 || T1 >: T2
    T = T1 <: T2 ? T1 : T2
    z = Variable{T}(ᵛ(x) .* ᵛ(y), x.backprop || y.backprop)
    if z.backprop
        z.backward = function DotMulBackward()
            if need2computeδ!(x)
                δx = δ(z) .* ᵛ(y)
                δ(x) .+= unbcast(δx, ᵛ(x))
            end
            if need2computeδ!(y)
                δy = δ(z) .* ᵛ(x)
                δ(y) .+= unbcast(δy, ᵛ(y))
            end
            ifNotKeepδThenFreeδ!(z)
        end
        addchild(z, x)
        addchild(z, y)
    end
    return z
end


function broadcasted(::typeof(*), x::Variable{T1}, y::T2) where {T1,T2}
    @assert T1 <: T2 || T1 >: T2
    T = T1 <: T2 ? T1 : T2
    z = Variable{T}(ᵛ(x) .* y, x.backprop)
    if z.backprop
        z.backward = function DotMulBackward()
            if need2computeδ!(x)
                δx = δ(z) .* ᵛ(y)
                δ(x) .+= unbcast(δx, ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(z)
        end
        addchild(z, x)
    end
    return z
end


function broadcasted(::typeof(*), x::T1, y::Variable{T2}) where {T1,T2}
    @assert T1 <: T2 || T1 >: T2
    T = T1 <: T2 ? T1 : T2
    z = Variable{T}(x .* ᵛ(y), y.backprop)
    if z.backprop
        z.backward = function DotMulBackward()
            if need2computeδ!(y)
                δy = δ(z) .* ᵛ(x)
                δ(y) .+= unbcast(δy, ᵛ(y))
            end
            ifNotKeepδThenFreeδ!(z)
        end
        addchild(z, y)
    end
    return z
end


# z = x ./ y
function broadcasted(::typeof(/), x::Variable{T1}, y::Variable{T2}) where {T1,T2}
    @assert T1 <: T2 || T1 >: T2
    T = T1 <: T2 ? T1 : T2
    z = Variable{T}(ᵛ(x) ./ ᵛ(y), x.backprop || y.backprop)
    if z.backprop
        z.backward = function DotDivBackward()
            δx = δ(z) ./ ᵛ(y)
            if need2computeδ!(x)
                δ(x) .+= unbcast(δx, ᵛ(x))
            end
            if need2computeδ!(y)
                δy = - δx .* ᵛ(z)
                δ(y) .+= unbcast(δy, ᵛ(y))
            end
            ifNotKeepδThenFreeδ!(z)
        end
        addchild(z, x)
        addchild(z, y)
    end
    return z
end


function broadcasted(::typeof(/), x::Variable{T1}, y::T2) where {T1,T2}
    @assert T1 <: T2 || T1 >: T2
    T = T1 <: T2 ? T1 : T2
    z = Variable{T}(ᵛ(x) ./ y, x.backprop)
    if z.backprop
        z.backward = function DotDivBackward()
            δx = δ(z) ./ ᵛ(y)
            if need2computeδ!(x)
                δ(x) .+= unbcast(δx, ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(z)
        end
        addchild(z, x)
    end
    return z
end


function broadcasted(::typeof(/), x::T1, y::Variable{T2}) where {T1,T2}
    @assert T1 <: T2 || T1 >: T2
    T = T1 <: T2 ? T1 : T2
    z = Variable{T}(x ./ ᵛ(y), y.backprop)
    if z.backprop
        z.backward = function DotDivBackward()
            if need2computeδ!(y)
                δy = - δ(z) ./ ᵛ(y) .* ᵛ(z)
                δ(y) .+= unbcast(δy, ᵛ(y))
            end
            ifNotKeepδThenFreeδ!(z)
        end
        addchild(z, y)
    end
    return z
end
