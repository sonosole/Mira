export Conv1d


"""
Applies a 1-D convolution over an 3-D input tensor of shape (ichannels, steps, batchsize)\n

# Constructor
    Conv1d(ichannels::Int, ochannels::Int, fn::FunOrNil=relu;
           kernel   :: Int = 3,
           dilation :: Int = 1,
           stride   :: Int = 1,
           padval   :: Real = 0f0,
           padmode  :: String = "repeat",
           padding  :: Dims2OrStr = "valid",
           type     :: Type = Array{Float32})
"""
mutable struct Conv1d <: Block
    w :: VarOrNil
    b :: VarOrNil
    f :: FunOrNil
    kernel   :: Dims{1}
    dilation :: Dims{1}
    stride   :: Dims{1}
    padding  :: Pads{1}
    padmode  :: Function
    padval   :: Float32
    function Conv1d(ichannels::Int, ochannels::Int, fn::FunOrNil=relu;
                    kernel   :: Int = 3,
                    dilation :: Int = 1,
                    stride   :: Int = 1,
                    padval   :: Real = 0f0,
                    padmode  :: String = "repeat",
                    padding  :: Dims2OrStr = "valid",
                    type     :: Type = Array{Float32})

        if padding isa String
            padding = inferpadding(padding, kernel, stride, dilation)
        else
            padding = singletuple(padding)
        end

        kernel   = singletuple(kernel)
        dilation = singletuple(dilation)
        stride   = singletuple(stride)
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
            padding, selectpad(padmode), padval)
    end
    function Conv1d()
        new(nothing, nothing, nothing, 3, 1, 1, ((0,0),), padzeros, 0f0)
    end
end


function clone(this::Conv1d; type::Type=Array{Float32})
    cloned   = Conv1d()
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
function Base.show(io::IO, m::Conv1d)
    P = ifelse(paddings(m.padding)==0, "", " padding=$(first(m.padding)),")
    D = ifelse(first(m.dilation)==1,   "", " dilation=$(first(m.dilation)),")
    S = ifelse(first(m.stride)==1,     "", " stride=$(first(m.stride)),")
    SIZE = size(m.w)
    TYPE = typeof(m.w.value)
    och  = SIZE[2] ÷ prod(m.kernel)
    ich  = SIZE[1]
    print(io, "Conv1d($och => $ich, $(m.f), kernel=$(first(m.kernel)),$D$S$P type=$TYPE)")
end



function paramsof(m::Conv1d)
    params = Vector{Variable}(undef,2)
    params[1] = m.w
    params[2] = m.b
    return params
end


function xparamsof(m::Conv1d)
    xparams = Vector{XVariable}(undef,2)
    xparams[1] = ('w', m.w)
    xparams[2] = ('b', m.b)
    return xparams
end


function nparamsof(m::Conv1d)
    lw = length(m.w)
    lb = length(m.b)
    return (lw + lb)
end

elsizeof(c::Conv1d) = elsizeof(c.w)

function bytesof(model::Conv1d, unit::String="MB")
    n = nparamsof(model) * elsizeof(model.w)
    return blocksize(n, uppercase(unit))
end



function forward(C::Conv1d, x::Variable, backend::Function=ten2mat)
    w = C.w
    b = C.b
    S = fullsize(w, x, C.padding, C.kernel, C.dilation, C.stride)
    y = backend(    x, C.padding, C.kernel, C.dilation, C.stride, C.padmode, C.padval)
    z = reshape(matAddVec(w * y, b), S)
    return C.f(z)
end


function predict(C::Conv1d, x::AbstractArray, backend::Function=ten2mat)
    w = value(C.w)
    b = value(C.b)
    S = fullsize(w, x, C.padding, C.kernel, C.dilation, C.stride)
    y = backend(    x, C.padding, C.kernel, C.dilation, C.stride, C.padmode, C.padval)
    z = reshape(w * y .+ b, S)
    return C.f(z)
end
