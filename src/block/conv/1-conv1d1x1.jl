export Conv1d1x1

"""
    mutable struct Conv1d1x1 <: Block

Applies y = fn(w * x .+ b) transformation to the incoming data x
"""
mutable struct Conv1d1x1 <: Block
    w::VarOrNil
    b::VarOrNil
    f::Function
    function Conv1d1x1(isize::Int, osize::Int, fn::Function=relu; type::Type=Array{Float32})
        T = eltype(type)
        A = sqrt(T(2/isize))
        w = randn(T, osize, isize) .* A
        b = randn(T, osize,     1) .* A
        new(Variable{type}(w,true,true,true), Variable{type}(b,true,true,true), fn)
    end
    function Conv1d1x1(fn::Function)
        new(nothing, nothing, fn)
    end
end


function clone(this::Conv1d1x1; type::Type=Array{Float32})
    cloned = Conv1d1x1(this.f)
    cloned.w = clone(this.w, type=type)
    cloned.b = clone(this.b, type=type)
    return cloned
end


# pretty show
function Base.show(io::IO, m::Conv1d1x1)
    SIZE = size(m.w)
    TYPE = typeof(m.w.value)
    print(io, "Conv1d1x1($(SIZE[2]), $(SIZE[1]), $(m.f); type=$TYPE)")
end


function forward(m::Conv1d1x1, x::Variable)
    f = m.f
    w = m.w
    b = m.b

    C, T, B = size(x)
    x = reshape(x, (C, T*B))
    y = f( matAddVec(w*x, b) )
    C = size(y, 1)
    return reshape(y, (C, T, B))
end



function predict(m::Conv1d1x1, x)
    f = m.f
    w = value(m.w)
    b = value(m.b)

    C, T, B = size(x)
    x = reshape(x, (C, T*B))
    y = f(w * x .+ b)
    C = size(y, 1)
    return reshape(y, (C, T, B))
end



"""
    unbiasedof(m::Conv1d1x1)

unbiased weights of Conv1d1x1 block
"""
function unbiasedof(m::Conv1d1x1)
    weights = Vector(undef, 1)
    weights[1] = m.w.value
    return weights
end


function weightsof(m::Conv1d1x1)
    weights = Vector(undef, 2)
    weights[1] = m.w.value
    weights[2] = m.b.value
    return weights
end


function gradsof(m::Conv1d1x1)
    grads = Vector(undef, 2)
    grads[1] = m.w.delta
    grads[2] = m.b.delta
    return grads
end


function zerograds!(m::Conv1d1x1)
    for v in gradsof(m)
        v .= 0.0
    end
end


function paramsof(m::Conv1d1x1)
    params = Vector{Variable}(undef,2)
    params[1] = m.w
    params[2] = m.b
    return params
end


function xparamsof(m::Conv1d1x1)
    xparams = Vector{XVariable}(undef,2)
    xparams[1] = ('w', m.w)
    xparams[2] = ('b', m.b)
    return xparams
end


function nparamsof(m::Conv1d1x1)
    lw = length(m.w)
    lb = length(m.b)
    return (lw + lb)
end

elsizeof(d::Conv1d1x1) = elsizeof(d.w)

function bytesof(model::Conv1d1x1, unit::String="MB")
    n = nparamsof(model) * elsizeof(model)
    return blocksize(n, uppercase(unit))
end


function nops(d::Conv1d1x1)
    m, n = size(d.w)
    mops = m * n
    aops = m * (n-1) + m
    acts = m
    return (mops, aops, acts) # (mul, add, act)
end
