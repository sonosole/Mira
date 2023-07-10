export Conv2d


"""
    mutable struct Conv2d <: Block

Applies a 2-D convolution over an 4-D input tensor of shape (ichannels, hight, width, batchsize)\n
"""
mutable struct Conv2d <: Block
    w :: VarOrNil
    b :: VarOrNil
    f :: FunOrNil
    kernel   :: Dims{2}
    dilation :: Dims{2}
    stride   :: Dims{2}
    padding  :: Pads{2}
    padmode  :: Function
    padval   :: Float32
    function Conv2d(ichannels::Int, ochannels::Int, fn::FunOrNil=relu;
                    kernel   :: Dims{2} = (3,3),
                    dilation :: Dims{2} = (1,1),
                    stride   :: Dims{2} = (1,1),
                    padval   :: Real = 0f0,
                    padmode  :: String  = "zeros",
                    padding  :: Pads2OrStr = "valid",
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
    function Conv2d()
        new(nothing, nothing, nothing, (3,3), (1,1), (1,1), ((0,0),(0,0)), padzeros, 0f0)
    end
end


function clone(this::Conv2d; type::Type=Array{Float32})
    cloned   = Conv2d()
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
function Base.show(io::IO, m::Conv2d)
    SIZE = size(m.w)
    TYPE = typeof(m.w.value)
    P = ifelse(paddings(m.padding)==0, "", " padding=$(m.padding),")
    D = ifelse(prod(m.dilation)==1,   "", " dilation=$(m.dilation),")
    S = ifelse(prod(m.stride)==1,     "", " stride=$(m.stride),")
    print(io, "Conv2d($(Int(SIZE[2]/prod(m.kernel))) => $(SIZE[1]), $(m.f), kernel=$(m.kernel),$D$S$P type=$TYPE)")
end


"""
    unbiasedof(m::Conv2d)

unbiased weights of Conv2d block
"""
function unbiasedof(m::Conv2d)
    weights = Vector(undef, 1)
    weights[1] = m.w.value
    return weights
end


function weightsof(m::Conv2d)
    weights = Vector(undef, 2)
    weights[1] = m.w.value
    weights[2] = m.b.value
    return weights
end


function gradsof(m::Conv2d)
    grads = Vector(undef, 2)
    grads[1] = m.w.delta
    grads[2] = m.b.delta
    return grads
end


function zerograds!(m::Conv2d)
    for v in gradsof(m)
        v .= 0.0
    end
end


function paramsof(m::Conv2d)
    params = Vector{Variable}(undef,2)
    params[1] = m.w
    params[2] = m.b
    return params
end


function xparamsof(m::Conv2d)
    xparams = Vector{XVariable}(undef,2)
    xparams[1] = ('w', m.w)
    xparams[2] = ('b', m.b)
    return xparams
end


function nparamsof(m::Conv2d)
    lw = length(m.w)
    lb = length(m.b)
    return (lw + lb)
end

elsizeof(c::Conv2d) = elsizeof(c.w)

function bytesof(model::Conv2d, unit::String="MB")
    n = nparamsof(model) * elsizeof(model.w)
    return blocksize(n, uppercase(unit))
end



function forward(C::Conv2d, x::Variable, backend::Function=ten2mat)
    w = C.w
    b = C.b
    S = fullsize(w, x, C.padding, C.kernel, C.dilation, C.stride)
    y = backend(    x, C.padding, C.kernel, C.dilation, C.stride, C.padmode, C.padval)
    z = reshape(matAddVec(w * y, b), S)
    return C.f(z)
end


function predict(C::Conv2d, x::AbstractArray, backend::Function=ten2mat)
    w = value(C.w)
    b = value(C.b)
    S = fullsize(w, x, C.padding, C.kernel, C.dilation, C.stride)
    y = backend(    x, C.padding, C.kernel, C.dilation, C.stride, C.padmode, C.padval)
    z = reshape(w * y .+ b, S)
    return C.f(z)
end
