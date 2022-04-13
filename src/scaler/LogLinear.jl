"""
    mutable struct LogLinear <: Block

Applies a linear and log transformation to the incoming data:
y = log(w .* x .+ b) where w and b are 0-dim tensors
"""
mutable struct LogLinear <: Block
    w::VarOrNil
    b::VarOrNil
    function LogLinear(; type::Type=Array{Float32})
        T = eltype(type)
        w = zeros(T, 1)
        b = zeros(T, 1)
        new(Variable{type}(w,true,true,true),
            Variable{type}(b,true,true,true))
    end
    function LogLinear()
        new(nothing, nothing)
    end
end

function clone(this::LogLinear; type::Type=Array{Float32})
    cloned = LogLinear()
    cloned.w = clone(this.w, type=type)
    cloned.b = clone(this.b, type=type)
    return cloned
end


# pretty show
function Base.show(io::IO, m::LogLinear)
    SIZE = size(m.w)
    TYPE = typeof(m.w.value)
    print(io, "Linear($(m.w[1]), $(m.b[1]); type=$TYPE)")
end



"""
    unbiasedof(m::LogLinear)

unbiased weights of `LogLinear` block
"""
function unbiasedof(m::LogLinear)
    weights = Vector(undef, 1)
    weights[1] = m.w.value
    return weights
end

function weightsof(m::LogLinear)
    weights = Vector(undef, 2)
    weights[1] = m.w.value
    weights[2] = m.b.value
    return weights
end


function gradsof(m::LogLinear)
    grads = Vector(undef, 2)
    grads[1] = m.w.delta
    grads[2] = m.b.delta
    return grads
end


function zerograds!(m::LogLinear)
    for v in gradsof(m)
        v .= 0.0
    end
end


function paramsof(m::LogLinear)
    params = Vector{Variable}(undef,2)
    params[1] = m.w
    params[2] = m.b
    return params
end


function xparamsof(m::LogLinear)
    xparams = Vector{XVariable}(undef,2)
    xparams[1] = ('w', m.w)
    xparams[2] = ('b', m.b)
    return xparams
end


function nparamsof(m::LogLinear)
    return 2
end

elsizeof(l::LogLinear) = elsizeof(l.w)

function bytesof(model::LogLinear, unit::String="MB")
    n = nparamsof(model) * elsizeof(model)
    return blocksize(n, uppercase(unit))
end


function forward(m::LogLinear, x::Variable)
    w = m.w
    b = m.b
    c = log.(ᵛ(w) .* ᵛ(x) .+ ᵛ(b))
    y = Variable{T}(c, x.backprop)
    if y.backprop
        if w.value[1] < 0.0f0
            w.value[1] = 1e-3
        end
        if b.value[1] < 0.0f0
            b.value[1] = 1e-3
        end
        y.backward = function LogLinearBackward()
            c⁻¹ = 1 ./ c
            if need2computeδ!(w)
                δ(w) .+= ᵛ(x) .* c⁻¹
            end
            if need2computeδ!(b)
                δ(b) .+= c⁻¹
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
        addchild(y, w)
        addchild(y, b)
    end
    return y
end


function predict(m::LogLinear, x)
    w = m.w.value
    b = m.b.value
    return log.(w .* x .+ b)
end


function nops(l::LogLinear)
    @info "ops of LogLinear depends on the length of input, so it can't be inffered"
    return (0, 0, 0) # (mul, add, act)
end
