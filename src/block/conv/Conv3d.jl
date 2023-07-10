export Conv3d


"""
    mutable struct Conv3d <: Block

Applies a 3-D convolution over an 5-D input tensor of shape (ichannels, hight, width, steps, batchsize)\n
"""
mutable struct Conv3d <: Block
    w :: VarOrNil
    b :: VarOrNil
    f :: FunOrNil
    kernel   :: Dims{3}
    dilation :: Dims{3}
    stride   :: Dims{3}
    padding  :: Pads{3}
    padmode  :: Function
    padval   :: Float32
    function Conv3d(ichannels::Int, ochannels::Int, fn::FunOrNil=relu;
                    kernel   :: Dims{3} = (3,3,3),
                    dilation :: Dims{3} = (1,1,1),
                    stride   :: Dims{3} = (1,1,1),
                    padval   :: Real = 0f0,
                    padmode  :: String  = "zeros",
                    padding  :: Pads3OrStr = "valid",
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
    function Conv3d()
        new(nothing, nothing, nothing, (3,3,3), (1,1,1), (1,1,1), ((0,0),(0,0),(0,0)), padzeros, 0f0)
    end
end


function clone(this::Conv3d; type::Type=Array{Float32})
    cloned   = Conv3d()
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
function Base.show(io::IO, m::Conv3d)
    SIZE = size(m.w)
    TYPE = typeof(m.w.value)
    P = ifelse(paddings(m.padding)==0, "", " padding=$(m.padding),")
    D = ifelse(prod(m.dilation)==1,   "", " dilation=$(m.dilation),")
    S = ifelse(prod(m.stride)==1,     "", " stride=$(m.stride),")
    print(io, "Conv3d($(Int(SIZE[2]/prod(m.kernel))) => $(SIZE[1]), $(m.f), kernel=$(m.kernel),$D$S$P type=$TYPE)")
end


"""
    unbiasedof(m::Conv3d)

unbiased weights of Conv3d block
"""
function unbiasedof(m::Conv3d)
    weights = Vector(undef, 1)
    weights[1] = m.w.value
    return weights
end


function weightsof(m::Conv3d)
    weights = Vector(undef, 2)
    weights[1] = m.w.value
    weights[2] = m.b.value
    return weights
end


function gradsof(m::Conv3d)
    grads = Vector(undef, 2)
    grads[1] = m.w.delta
    grads[2] = m.b.delta
    return grads
end


function zerograds!(m::Conv3d)
    for v in gradsof(m)
        v .= 0.0
    end
end


function paramsof(m::Conv3d)
    params = Vector{Variable}(undef,2)
    params[1] = m.w
    params[2] = m.b
    return params
end


function xparamsof(m::Conv3d)
    xparams = Vector{XVariable}(undef,2)
    xparams[1] = ('w', m.w)
    xparams[2] = ('b', m.b)
    return xparams
end


function nparamsof(m::Conv3d)
    lw = length(m.w)
    lb = length(m.b)
    return (lw + lb)
end

elsizeof(c::Conv3d) = elsizeof(c.w)

function bytesof(model::Conv3d, unit::String="MB")
    n = nparamsof(model) * elsizeof(model.w)
    return blocksize(n, uppercase(unit))
end



function forward(C::Conv3d, x::Variable, backend::Function=ten2mat)
    w = C.w
    b = C.b
    S = fullsize(w, x, C.padding, C.kernel, C.dilation, C.stride)
    y = backend(    x, C.padding, C.kernel, C.dilation, C.stride, C.padmode, C.padval)
    z = reshape(matAddVec(w * y, b), S)
    return C.f(z)
end


function predict(C::Conv3d, x::AbstractArray, backend::Function=ten2mat)
    w = value(C.w)
    b = value(C.b)
    S = fullsize(w, x, C.padding, C.kernel, C.dilation, C.stride)
    y = backend(    x, C.padding, C.kernel, C.dilation, C.stride, C.padmode, C.padval)
    z = reshape(w * y .+ b, S)
    return C.f(z)
end
