export Conv
export Conv1d
export Conv2d
export Conv3d
export Conv4d
export Conv5d


"""
Applies a `D`-dim convolution over an `(D+2)`-dim input tensor of shape (ichannels, w1, w2, ..., wn, batchsize)\n
# Constructor
    Conv{D}(ichannels::Int, ochannels::Int, fn::FunOrNil=relu;
            kernel   :: Dims{D} = ntuple(i -> 3, D),
            dilation :: Dims{D} = ntuple(i -> 1, D),
            stride   :: Dims{D} = ntuple(i -> 1, D),
            padval   :: Real = 0f0,
            padmode  :: String  = "zeros",
            padding  :: PadsDOrStr = "valid",
            type     :: Type = Array{Float32}) where D

+ `padmode` should be one of \"zeros\", \"constant\", \"repeat\", \"reflect\", \"symmetric\", \"circular\"
+ `padding` can be \"valid\", \"same\", or type `NTuple{D, Dims{2}}`
"""
mutable struct Conv{D} <: Block
    w :: VarOrNil
    b :: VarOrNil
    f :: FunOrNil
    kernel   :: Dims{D}
    dilation :: Dims{D}
    stride   :: Dims{D}
    padding  :: Pads{D}
    padmode  :: Function
    padval   :: Float32
    function Conv{D}(ichannels::Int, ochannels::Int, fn::FunOrNil=relu;
                     kernel   :: Dims{D} = ntuple(i -> 3, D),
                     dilation :: Dims{D} = ntuple(i -> 1, D),
                     stride   :: Dims{D} = ntuple(i -> 1, D),
                     padval   :: Real = 0f0,
                     padmode  :: String  = "zeros",
                     padding  :: PadsDOrStr = "valid",
                     type     :: Type = Array{Float32}) where D

        if padding isa String
            padding = inferpadding(padding, kernel, stride, dilation)
        end

        dtype    = eltype(type)
        patchlen = prod(kernel) * ichannels
        Amplifer = dtype(sqrt(2 / patchlen))

        w = Amplifer * randn(dtype, ochannels, patchlen)
        b = Amplifer * randn(dtype, ochannels,        1)

        new{D}(Variable{type}(w,true,true,true),
               Variable{type}(b,true,true,true), fn,
               kernel, dilation, stride,
               padding, selectpad(padmode), padval)
    end
    function Conv{D}() where D
        O = (0, 0)
        new{D}(nothing, nothing, nothing,
               ntuple(i -> 3, D),
               ntuple(i -> 1, D),
               ntuple(i -> 1, D),
               ntuple(i -> O, D), padconst, 0f0)
    end
end


function clone(this::Conv{D}; type::Type=Array{Float32}) where D
    cloned   = Conv{D}()
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
function Base.show(io::IO, c::Conv{1})
    P = ifelse(paddings(c.padding)==0, "", " padding=$(first(c.padding)),")
    D = ifelse(first(c.dilation)==1,   "", " dilation=$(first(c.dilation)),")
    S = ifelse(first(c.stride)==1,     "", " stride=$(first(c.stride)),")
    SIZE = size(c.w)
    TYPE = typeof(c.w.value)
    och  = SIZE[2] ÷ prod(c.kernel)
    ich  = SIZE[1]
    print(io, "Conv1d($och => $ich, $(c.f), kernel=$(first(c.kernel)),$D$S$P type=$TYPE)")
end

function Base.show(io::IO, c::Conv{N}) where N
    P = ifelse(paddings(c.padding)==0, "", " padding=$(c.padding),")
    D = ifelse(prod(c.dilation)==1,   "", " dilation=$(c.dilation),")
    S = ifelse(prod(c.stride)==1,     "", " stride=$(c.stride),")
    SIZE = size(c.w)
    TYPE = typeof(c.w.value)
    och  = SIZE[2] ÷ prod(c.kernel)
    ich  = SIZE[1]
    print(io, "Conv$(N)d($(och) => $(ich), $(c.f), kernel=$(c.kernel),$D$S$P type=$TYPE)")
end


function paramsof(c::Conv{D}) where D
    params = Vector{Variable}(undef,2)
    params[1] = c.w
    params[2] = c.b
    return params
end


function xparamsof(c::Conv{D}) where D
    xparams = Vector{XVariable}(undef,2)
    xparams[1] = ('w', c.w)
    xparams[2] = ('b', c.b)
    return xparams
end


function nparamsof(c::Conv{D}) where D
    lw = length(c.w)
    lb = length(c.b)
    return (lw + lb)
end

elsizeof(c::Conv{D}) where D = elsizeof(c.w)

function bytesof(c::Conv{D}, unit::String="MB") where D
    n = nparamsof(c) * elsizeof(c)
    return blocksize(n, uppercase(unit))
end



function forward(C::Conv{D}, x::Variable, backend::Function=ten2mat) where D
    w = C.w
    b = C.b
    S = fullsize(w, x, C.padding, C.kernel, C.dilation, C.stride)
    y = backend(    x, C.padding, C.kernel, C.dilation, C.stride, C.padmode, C.padval)
    z = reshape(matAddVec(w * y, b), S)
    return C.f(z)
end


function predict(C::Conv{D}, x::AbstractArray, backend::Function=ten2mat) where D
    w = value(C.w)
    b = value(C.b)
    S = fullsize(w, x, C.padding, C.kernel, C.dilation, C.stride)
    y = backend(    x, C.padding, C.kernel, C.dilation, C.stride, C.padmode, C.padval)
    z = reshape(w * y .+ b, S)
    return C.f(z)
end



"""
Applies a `1`-D convolution over an `3`-D input tensor of shape (ichannels, `steps`, batchsize)\n

# Constructor
    Conv1d(ichannels::Int, ochannels::Int, fn::FunOrNil=relu;
           kernel   :: Int = 3,
           dilation :: Int = 1,
           stride   :: Int = 1,
           padval   :: Real = 0f0,
           padmode  :: String = "repeat",
           padding  :: Dims2OrStr = "valid",
           type     :: Type = Array{Float32})

+ `padmode` should be one of \"zeros\", \"constant\", \"repeat\", \"reflect\", \"symmetric\", \"circular\"
+ `padding` can be \"valid\", \"same\", or type `Dims{2}`
"""
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

    return Conv{1}(ichannels, ochannels, fn;
                   kernel   = kernel,
                   dilation = dilation,
                   stride   = stride,
                   padval   = padval,
                   padmode  = padmode,
                   padding  = padding,
                   type     = type)
end


"""
Applies a `2`-D convolution over an `4`-D input tensor of shape (ichannels, `hight`, `width`, batchsize)\n

# Constructor
    Conv2d(ichannels::Int, ochannels::Int, fn::FunOrNil=relu;
           kernel   :: Dims{2} = (3,3),
           dilation :: Dims{2} = (1,1),
           stride   :: Dims{2} = (1,1),
           padval   :: Real = 0f0,
           padmode  :: String = "repeat",
           padding  :: Pads2OrStr = "valid",
           type     :: Type = Array{Float32})

+ `padmode` should be one of \"zeros\", \"constant\", \"repeat\", \"reflect\", \"symmetric\", \"circular\"
+ `padding` can be \"valid\", \"same\", or type `NTuple{2, Dims{2}}`
"""
function Conv2d(ichannels::Int, ochannels::Int, fn::FunOrNil=relu;
                kernel   :: Dims{2} = (3,3),
                dilation :: Dims{2} = (1,1),
                stride   :: Dims{2} = (1,1),
                padval   :: Real = 0f0,
                padmode  :: String = "repeat",
                padding  :: Pads2OrStr = "valid",
                type     :: Type = Array{Float32})

    return Conv{2}(ichannels, ochannels, fn;
                   kernel   = kernel,
                   dilation = dilation,
                   stride   = stride,
                   padval   = padval,
                   padmode  = padmode,
                   padding  = padding,
                   type     = type)
end



"""
Applies a `3`-D convolution over an `5`-D input tensor of shape (ichannels, `hight`, `width`, `steps`, batchsize)\n

# Constructor
    Conv3d(ichannels::Int, ochannels::Int, fn::FunOrNil=relu;
           kernel   :: Dims{3} = (3,3,3),
           dilation :: Dims{3} = (1,1,1),
           stride   :: Dims{3} = (1,1,1),
           padval   :: Real = 0f0,
           padmode  :: String = "repeat",
           padding  :: Pads3OrStr = "valid",
           type     :: Type = Array{Float32})

+ `padmode` should be one of \"zeros\", \"constant\", \"repeat\", \"reflect\", \"symmetric\", \"circular\"
+ `padding` can be \"valid\", \"same\", or type `NTuple{3, Dims{2}}`
"""
function Conv3d(ichannels::Int, ochannels::Int, fn::FunOrNil=relu;
                kernel   :: Dims{3} = (3,3,3),
                dilation :: Dims{3} = (1,1,1),
                stride   :: Dims{3} = (1,1,1),
                padval   :: Real = 0f0,
                padmode  :: String = "repeat",
                padding  :: Pads3OrStr = "valid",
                type     :: Type = Array{Float32})

    return Conv{3}(ichannels, ochannels, fn;
                   kernel   = kernel,
                   dilation = dilation,
                   stride   = stride,
                   padval   = padval,
                   padmode  = padmode,
                   padding  = padding,
                   type     = type)
end



"""
Applies a `4`-D convolution over an `6`-D input tensor of shape (ichannels, `w1`,`w2`,`w3`,`w4`, batchsize)\n

# Constructor
    Conv4d(ichannels::Int, ochannels::Int, fn::FunOrNil=relu;
           kernel   :: Dims{4} = (3,3,3,3),
           dilation :: Dims{4} = (1,1,1,1),
           stride   :: Dims{4} = (1,1,1,1),
           padval   :: Real = 0f0,
           padmode  :: String = "repeat",
           padding  :: Pads4OrStr = "valid",
           type     :: Type = Array{Float32})

+ `padmode` should be one of \"zeros\", \"constant\", \"repeat\", \"reflect\", \"symmetric\", \"circular\"
+ `padding` can be \"valid\", \"same\", or type `NTuple{4, Dims{2}}`
"""
function Conv4d(ichannels::Int, ochannels::Int, fn::FunOrNil=relu;
                kernel   :: Dims{4} = (3,3,3,3),
                dilation :: Dims{4} = (1,1,1,1),
                stride   :: Dims{4} = (1,1,1,1),
                padval   :: Real = 0f0,
                padmode  :: String = "repeat",
                padding  :: Pads4OrStr = "valid",
                type     :: Type = Array{Float32})

    return Conv{4}(ichannels, ochannels, fn;
                   kernel   = kernel,
                   dilation = dilation,
                   stride   = stride,
                   padval   = padval,
                   padmode  = padmode,
                   padding  = padding,
                   type     = type)
end


"""
Applies a `5`-D convolution over an `7`-D input tensor of shape (ichannels, `w1`,`w2`,`w3`,`w4`,`w5`, batchsize)\n

# Constructor
    Conv5d(ichannels::Int, ochannels::Int, fn::FunOrNil=relu;
           kernel   :: Dims{5} = (3,3,3,3,3),
           dilation :: Dims{5} = (1,1,1,1,1),
           stride   :: Dims{5} = (1,1,1,1,1),
           padval   :: Real = 0f0,
           padmode  :: String = "repeat",
           padding  :: Pads5OrStr = "valid",
           type     :: Type = Array{Float32})

+ `padmode` should be one of \"zeros\", \"constant\", \"repeat\", \"reflect\", \"symmetric\", \"circular\"
+ `padding` can be \"valid\", \"same\", or type `NTuple{5, Dims{2}}`
"""
function Conv5d(ichannels::Int, ochannels::Int, fn::FunOrNil=relu;
                kernel   :: Dims{5} = (3,3,3,3,3),
                dilation :: Dims{5} = (1,1,1,1,1),
                stride   :: Dims{5} = (1,1,1,1,1),
                padval   :: Real = 0f0,
                padmode  :: String = "repeat",
                padding  :: Pads5OrStr = "valid",
                type     :: Type = Array{Float32})

    return Conv{5}(ichannels, ochannels, fn;
                   kernel   = kernel,
                   dilation = dilation,
                   stride   = stride,
                   padval   = padval,
                   padmode  = padmode,
                   padding  = padding,
                   type     = type)
end