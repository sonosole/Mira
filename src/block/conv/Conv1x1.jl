export Conv1x1

"""
    mutable struct Conv1x1 <: Block

Applies y = fn(w * x .+ b) transformation to the incoming data x
# Constructor
    Conv1x1(ichannels, ochannels, fn::FunOrNil=relu; type = Array{Float32})
"""
mutable struct Conv1x1 <: Block
    w::VarOrNil
    b::VarOrNil
    f::FunOrNil
    function Conv1x1(ichannels::Int, ochannels::Int, fn::FunOrNil=relu; type::Type=Array{Float32})
        T = eltype(type)
        A = T( sqrt(2 / isize) )
        w = randn(T, ochannels, ichannels) .* A
        b = randn(T, ochannels,         1) .* A
        new(Variable{type}(w,true,true,true),
            Variable{type}(b,true,true,true), fn)
    end
    function Conv1x1(fn::FunOrNil)
        new(nothing, nothing, fn)
    end
end


function clone(this::Conv1x1; type::Type=Array{Float32})
    cloned   = Conv1x1(this.f)
    cloned.w = clone(this.w, type=type)
    cloned.b = clone(this.b, type=type)
    return cloned
end


# pretty show
function Base.show(io::IO, m::Conv1x1)
    SIZE = size(m.w)
    TYPE = typeof(m.w.value)
    print(io, "Conv1x1($(SIZE[2]), $(SIZE[1]), $(m.f); type=$TYPE)")
end


function forward(m::Conv1x1, x::Variable)
    f = m.f
    w = m.w
    b = m.b

    xsize = size(x)
    # -----------------------------------------------------
    D  = ndims(x)           # total dims of input/output
    L  = prod(xsize[2:D])   # total numbers of input/output
    Cᵢ = xsize[1]           # input  channels
    Cₒ = size(w,1)          # output channels
    # -----------------------------------------------------
    ysize = ntuple(i -> i>1 ? xsize[i] : Cₒ, D)

    x = reshape(x, Cᵢ, L)
    y = f( matAddVec(w*x, b) )

    return reshape(y, ysize)
end



function predict(m::Conv1x1, x::AbstractArray)
    f = m.f
    w = value(m.w)
    b = value(m.b)

    xsize = size(x)
    # -----------------------------------------------------
    D  = ndims(x)           # total dims of input/output
    L  = prod(xsize[2:D])   # total numbers of input/output
    Cᵢ = xsize[1]           # input  channels
    Cₒ = size(w,1)          # output channels
    # -----------------------------------------------------
    ysize = ntuple(i -> i>1 ? xsize[i] : Cₒ, D)

    x = reshape(x, Cᵢ, L)
    y = f( matAddVec(w*x, b) )

    return reshape(y, ysize)
end



"""
    unbiasedof(m::Conv1x1)

unbiased weights of Conv1x1 block
"""
function unbiasedof(m::Conv1x1)
    weights = Vector(undef, 1)
    weights[1] = m.w.value
    return weights
end


function weightsof(m::Conv1x1)
    weights = Vector(undef, 2)
    weights[1] = m.w.value
    weights[2] = m.b.value
    return weights
end


function gradsof(m::Conv1x1)
    grads = Vector(undef, 2)
    grads[1] = m.w.delta
    grads[2] = m.b.delta
    return grads
end


function zerograds!(m::Conv1x1)
    for v in gradsof(m)
        v .= 0.0
    end
end


function paramsof(m::Conv1x1)
    params = Vector{Variable}(undef,2)
    params[1] = m.w
    params[2] = m.b
    return params
end


function xparamsof(m::Conv1x1)
    xparams = Vector{XVariable}(undef,2)
    xparams[1] = ('w', m.w)
    xparams[2] = ('b', m.b)
    return xparams
end


function nparamsof(m::Conv1x1)
    lw = length(m.w)
    lb = length(m.b)
    return (lw + lb)
end

elsizeof(d::Conv1x1) = elsizeof(d.w)

function bytesof(model::Conv1x1, unit::String="MB")
    n = nparamsof(model) * elsizeof(model)
    return blocksize(n, uppercase(unit))
end


function nops(d::Conv1x1, c::Int=1)
    m, n = size(d.w)
    mops = m * n
    aops = m * (n-1) + m
    acts = m
    return c .* (mops, aops, acts) # (mul, add, act)
end
