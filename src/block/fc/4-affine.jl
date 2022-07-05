"""
    mutable struct Affine <: Block

Applies a linear transformation to the incoming data: y = w * x
"""
mutable struct Affine <: Block
    w::VarOrNil # input to hidden weights
    function Affine(isize::Int, hsize::Int; type::Type=Array{Float32})
        T = eltype(type)
        A = T(sqrt(1 / isize))
        w = randn(T, hsize, isize) .* A
        new(Variable{type}(w,true,true,true))
    end
    function Affine()
        new(nothing)
    end
end


function clone(this::Affine; type::Type=Array{Float32})
    cloned = Affine()
    cloned.w = clone(this.w, type=type)
    return cloned
end


# pretty show
function Base.show(io::IO, m::Affine)
    SIZE = size(m.w)
    TYPE = typeof(m.w.value)
    print(io, "Affine($(SIZE[2]), $(SIZE[1]); type=$TYPE)")
end

"""
    unbiasedof(m::Affine)

unbiased weights of Affine block
"""
function unbiasedof(m::Affine)
    weights = Vector(undef, 1)
    weights[1] = m.w.value
    return weights
end

function weightsof(m::Affine)
    weights = Vector(undef, 1)
    weights[1] = m.w.value
    return weights
end


function gradsof(m::Affine)
    grads = Vector(undef, 1)
    grads[1] = m.w.delta
    return grads
end


function zerograds!(m::Affine)
    for v in gradsof(m)
        v .= 0.0
    end
end


function paramsof(m::Affine)
    params = Vector{Variable}(undef,1)
    params[1] = m.w
    return params
end


function xparamsof(m::Affine)
    xparams = Vector{XVariable}(undef,1)
    xparams[1] = ('w', m.w)
    return xparams
end


function nparamsof(m::Affine)
    return length(m.w)
end

elsizeof(a::Affine) = elsizeof(a.w)

function bytesof(model::Affine, unit::String="MB")
    n = nparamsof(model) * elsizeof(model)
    return blocksize(n, uppercase(unit))
end


function forward(m::Affine, x::Variable)
    w = m.w
    return w * x
end


function predict(m::Affine, x)
    w = ᵛ(m.w)
    return w * x
end


function to(type::Type, m::Affine)
    m.w = to(type, m.w)
    return m
end


function to!(type::Type, m::Affine)
    m.w = to(type, m.w)
    return nothing
end


function nops(a::Affine)
    m, n = size(a.w)
    mops = m * n
    aops = m * (n-1)
    acts = 0
    return (mops, aops, acts)
end
