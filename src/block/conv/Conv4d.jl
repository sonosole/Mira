export Conv4d


"""
    mutable struct Conv4d <: Block

Applies a 4-D convolution over an 6-D input tensor of shape (ichannels, w1, w2, w3, w4, batchsize)\n
"""
mutable struct Conv4d <: Block
    w :: VarOrNil
    b :: VarOrNil
    f :: FunOrNil
    kernel   :: Dims{4}
    dilation :: Dims{4}
    stride   :: Dims{4}
    padding  :: Pads{4}
    padmode  :: Function
    padval   :: Float32
    function Conv4d(ichannels::Int, ochannels::Int, fn::FunOrNil=relu;
                    kernel   :: Dims{4} = (3,3,3,3),
                    dilation :: Dims{4} = (1,1,1,1),
                    stride   :: Dims{4} = (1,1,1,1),
                    padval   :: Real = 0f0,
                    padmode  :: String  = "zeros",
                    padding  :: Pads4OrStr = "valid",
                    type     :: Type = Array{Float32})

        if padding isa String
            padding = inferpadding(padding, kernel, stride, dilation)
        end

        dtype    = eltype(type)
        patchlen = prod(kernel) * ichannels
        Amplifer = dtype(sqrt(2 / patchlen))
        w = Amplifer * randn(dtype, ochannels, patchlen)
        b = Amplifer * randn(dtype, ochannels,        1)
        new(Variable{type}(w,true,true,true),
            Variable{type}(b,true,true,true), fn,
            kernel,
            dilation,
            stride,
            padding,
            selectpad(padmode), padval)
    end
    function Conv4d()
        new(nothing, nothing, nothing, (3,3,3,3), (1,1,1,1), (1,1,1,1), ((0,0),(0,0),(0,0),(0,0)), padzeros, 0f0)
    end
end


function clone(this::Conv4d; type::Type=Array{Float32})
    cloned   = Conv4d()
    cloned.w = clone(this.w, type=type)
    cloned.b = clone(this.b, type=type)
    cloned.f = this.f

    cloned.kernel   = this.kernel
    cloned.dilation = this.dilation
    cloned.stride   = this.stride
    cloned.padding  = this.padding
    cloned.padmode  = this.padmode
    cloned.padval   = this.padval
    return cloned
end


# pretty show
function Base.show(io::IO, m::Conv4d)
    SIZE = size(m.w)
    TYPE = typeof(m.w.value)
    P = ifelse(paddings(m.padding)==0, "", " padding=$(m.padding),")
    D = ifelse(prod(m.dilation)==1,   "", " dilation=$(m.dilation),")
    S = ifelse(prod(m.stride)==1,     "", " stride=$(m.stride),")
    print(io, "Conv4d($(Int(SIZE[2]/prod(m.kernel))) => $(SIZE[1]), $(m.f), kernel=$(m.kernel),$D$S$P type=$TYPE)")
end


"""
    unbiasedof(m::Conv4d)

unbiased weights of Conv4d block
"""
function unbiasedof(m::Conv4d)
    weights = Vector(undef, 1)
    weights[1] = m.w.value
    return weights
end


function weightsof(m::Conv4d)
    weights = Vector(undef, 2)
    weights[1] = m.w.value
    weights[2] = m.b.value
    return weights
end


function gradsof(m::Conv4d)
    grads = Vector(undef, 2)
    grads[1] = m.w.delta
    grads[2] = m.b.delta
    return grads
end


function zerograds!(m::Conv4d)
    for v in gradsof(m)
        v .= 0.0
    end
end


function paramsof(m::Conv4d)
    params = Vector{Variable}(undef,2)
    params[1] = m.w
    params[2] = m.b
    return params
end


function xparamsof(m::Conv4d)
    xparams = Vector{XVariable}(undef,2)
    xparams[1] = ('w', m.w)
    xparams[2] = ('b', m.b)
    return xparams
end


function nparamsof(m::Conv4d)
    lw = length(m.w)
    lb = length(m.b)
    return (lw + lb)
end

elsizeof(c::Conv4d) = elsizeof(c.w)

function bytesof(model::Conv4d, unit::String="MB")
    n = nparamsof(model) * elsizeof(model.w)
    return blocksize(n, uppercase(unit))
end



function forward(C::Conv4d, x::Variable, backend::Function=ten2mat)
    w = C.w
    b = C.b
    S = fullsize(w, x, C.padding, C.kernel, C.dilation, C.stride)
    y = backend(    x, C.padding, C.kernel, C.dilation, C.stride, C.padmode, C.padval)
    z = reshape(matAddVec(w * y, b), S)
    return C.f(z)
end


function predict(C::Conv4d, x::AbstractArray, backend::Function=ten2mat)
    w = value(C.w)
    b = value(C.b)
    S = fullsize(w, x, C.padding, C.kernel, C.dilation, C.stride)
    y = backend(    x, C.padding, C.kernel, C.dilation, C.stride, C.padmode, C.padval)
    z = reshape(w * y .+ b, S)
    return C.f(z)
end
