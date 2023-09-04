export TransConv
export TransConv1d
export TransConv2d
export TransConv3d
export TransConv4d
export TransConv5d
export setoutsize

"""
Applies a `D`-dim transpose convolution over an `(D+2)`-dim input tensor of shape (ichannels, w1, w2, ..., wn, batchsize)\n
# Constructor
    TransConv{D}(ichannels::Int, ochannels::Int, fn::FunOrNil=relu;
                 kernel   :: Dims{D} = ntuple(i -> 3, D),
                 dilation :: Dims{D} = ntuple(i -> 1, D),
                 stride   :: Dims{D} = ntuple(i -> 1, D),
                 padding  :: PadsDOrStr = "valid",
                 type     :: Type = Array{Float32}) where D
+ `ichannels` is the number of input's channels
+ `ochannels` is the number of output's channels
+ `padding` can be "valid", "same", or type `NTuple{D, Dims{2}}`
# Detailed Processes
The transpose convolution is X = TransConv(Z), decomposed into following:
```julia
   ┌──────────────────────────────────────────────────────────────────┐
   │ ┌────────────────────────┐         ┌───────────┐       ┌───────┐ │
X ←│ │[unpad] ← Xten ← [toten]│← Xmat ← │ W*(∙) + B │ ← Y ← │reshape│ │← Z
   │ └────────mat2ten─────────┘         └───Dense───┘       └───────┘ │
   └──────────────────────────────────────────────────────────────────┘
```
"""
mutable struct TransConv{D} <: Block
    w :: VarOrNil
    b :: VarOrNil
    f :: FunOrNil
    kernel   :: Dims{D}
    dilation :: Dims{D}
    stride   :: Dims{D}
    padding  :: Pads{D}
    outsize  :: DimsOrNil
    function TransConv{D}(ichannels::Int, ochannels::Int, fn::FunOrNil=relu;
                          kernel   :: Dims{D} = ntuple(i -> 3, D),
                          dilation :: Dims{D} = ntuple(i -> 1, D),
                          stride   :: Dims{D} = ntuple(i -> 1, D),
                          padding  :: PadsDOrStr = "valid",
                          type     :: Type = Array{Float32}) where D

        if padding isa String
            npads = inferpadding(padding, kernel, stride, dilation)
        elseif padding isa Int
            npads = ntuple(i -> (padding, padding), D)
        else
            npads = ntuple(D) do i
                if padding[i] isa Int
                    return (padding[i], padding[i])
                end
                if padding[i] isa Dims{2}
                    return padding[i]
                end
            end
        end

        dtype    = eltype(type)
        patchlen = prod(kernel) * ochannels
        Amplifer = dtype(sqrt(2 / patchlen))

        w = Amplifer * randn(dtype, patchlen, ichannels)
        b = Amplifer * randn(dtype, patchlen,        1)

        new{D}(Variable{type}(w,true,true,true),
               Variable{type}(b,true,true,true), fn,
               kernel, dilation, stride, npads, nothing)
    end
    function TransConv{D}() where D
        O = (0, 0)
        new{D}(nothing, nothing, nothing,
               ntuple(i -> 3, D),
               ntuple(i -> 1, D),
               ntuple(i -> 1, D),
               ntuple(i -> O, D), nothing)
    end
end


function clone(this::TransConv{D}; type::Type=Array{Float32}) where D
    cloned   = TransConv{D}()
    cloned.w = clone(this.w, type=type)
    cloned.b = clone(this.b, type=type)
    cloned.f = this.f

    cloned.kernel   = this.kernel
    cloned.dilation = this.dilation
    cloned.stride   = this.stride
    cloned.padding  = this.padding
    cloned.outsize  = this.outsize
    return cloned
end


function Base.show(io::IO, c::TransConv{N}) where N
    P = ifelse(paddings(c.padding)==0, "", " padding=$(c.padding),")
    D = ifelse(prod(c.dilation)==1,   "", " dilation=$(c.dilation),")
    S = ifelse(prod(c.stride)==1,     "", " stride=$(c.stride),")
    SIZE =   size(c.w.value)
    TYPE = typeof(c.w.value)
    och  = SIZE[1] ÷ prod(c.kernel)
    ich  = SIZE[2]
    print(io, "TransConv$(N)d($ich => $och, $(c.f), kernel=$(c.kernel),$D$S$P type=$TYPE)")
end


function Base.show(io::IO, c::TransConv{1})
    P = ifelse(paddings(c.padding)==0, "", " padding=$(first(c.padding)),")
    D = ifelse(first(c.dilation)==1,   "", " dilation=$(first(c.dilation)),")
    S = ifelse(first(c.stride)==1,     "", " stride=$(first(c.stride)),")
    SIZE =   size(c.w.value)
    TYPE = typeof(c.w.value)
    och  = SIZE[1] ÷ prod(c.kernel)
    ich  = SIZE[2]
    print(io, "TransConv1d($ich => $och, $(c.f), kernel=$(first(c.kernel)),$D$S$P type=$TYPE)")
end

function fan_in_out(c::TransConv)
    SIZE = size(c.w)
    och  = SIZE[1] ÷ prod(c.kernel)
    ich  = SIZE[2]
    return ich, och
end

function fanin(c::TransConv)
    SIZE = size(c.w)
    ich  = SIZE[2]
    return ich
end

function fanout(c::TransConv)
    SIZE = size(c.w)
    och  = SIZE[1] ÷ prod(c.kernel)
    return och
end


function setoutsize(C::TransConv{D}, sz::Dims{M}) where {D,M}
    @assert M==D+2 "TransConv$(D)d's dims of output shall be $(D+2) but got $M"
    C.outsize = sz
    return nothing
end


function forward(C::TransConv{D}, Z::Variable) where D
    W = C.w
    B = C.b
    N = D + 2

    ZSIZE = size(Z)
    YSIZE = (ZSIZE[1], prod(ZSIZE[2:N]))

    Y = reshape(Z, YSIZE)
    X = mat2ten(W * Y .+ B, ZSIZE, C.outsize, C.padding, C.kernel, C.dilation, C.stride)
    return C.f(X)
end


function predict(C::TransConv{D}, Z::AbstractArray) where D
    W = value(C.w)
    B = value(C.b)
    N = D + 2

    ZSIZE = size(Z)
    YSIZE = (ZSIZE[1], prod(ZSIZE[2:N]))

    Y = reshape(Z, YSIZE)
    X = mat2ten(W * Y .+ B, ZSIZE, C.outsize, C.padding, C.kernel, C.dilation, C.stride)
    return C.f(X)
end


function paramsof(c::TransConv)
    params = Vector{Variable}(undef,2)
    params[1] = c.w
    params[2] = c.b
    return params
end


function xparamsof(c::TransConv)
    xparams = Vector{XVariable}(undef,2)
    xparams[1] = ('w', c.w)
    xparams[2] = ('b', c.b)
    return xparams
end

function nparamsof(c::TransConv)
    lw = length(c.w)
    lb = length(c.b)
    return (lw + lb)
end

elsizeof(c::TransConv) = elsizeof(c.w)

function bytesof(c::TransConv, unit::String="MB")
    n = nparamsof(c) * elsizeof(c)
    return blocksize(n, uppercase(unit))
end


"""
Applies a `1`-D transpose convolution over an `3`-D input tensor of shape (ichannels, `steps`, batchsize)\n

# Constructor
    TransConv1d(ichannels::Int, ochannels::Int, fn::FunOrNil=relu;
                kernel   :: Int = 3,
                dilation :: Int = 1,
                stride   :: Int = 1,
                padding  :: Dims2OrStr = "valid",
                type     :: Type = Array{Float32})
+ `ichannels` is the number of input's channels
+ `ochannels` is the number of output's channels
+ `padding` can be "valid", "same", or type `Dims{2}`
"""
function TransConv1d(ichannels::Int, ochannels::Int, fn::FunOrNil=relu;
                     kernel   :: Int = 3,
                     dilation :: Int = 1,
                     stride   :: Int = 1,
                     padding  :: Pads1OrStr = "valid",
                     type     :: Type = Array{Float32})

    if !isa(padding, String)
        padding = singletuple(padding)
    end

    kernel   = singletuple(kernel)
    dilation = singletuple(dilation)
    stride   = singletuple(stride)

    return TransConv{1}(ichannels, ochannels, fn; kernel, dilation, stride, padding, type)
end


"""
Applies a `2`-D transpose convolution over an `4`-D input tensor of shape (ichannels, `hight`, `width`, batchsize)\n

# Constructor
    TransConv2d(ichannels::Int, ochannels::Int, fn::FunOrNil=relu;
                kernel   :: Dims{2} = (3,3),
                dilation :: Dims{2} = (1,1),
                stride   :: Dims{2} = (1,1),
                padding  :: Pads2OrStr = "valid",
                type     :: Type = Array{Float32})
+ `ichannels` is the number of input's channels
+ `ochannels` is the number of output's channels
+ `padding` can be "valid", "same", or type `NTuple{2, Dims{2}}`
"""
function TransConv2d(ichannels::Int, ochannels::Int, fn::FunOrNil=relu;
                     kernel   :: Dims{2} = (3,3),
                     dilation :: Dims{2} = (1,1),
                     stride   :: Dims{2} = (1,1),
                     padding  :: Pads2OrStr = "valid",
                     type     :: Type = Array{Float32})

    return TransConv{2}(ichannels, ochannels, fn; kernel, dilation, stride, padding, type)
end



"""
Applies a `3`-D transpose convolution over an `5`-D input tensor of shape (ichannels, `hight`, `width`, `steps`, batchsize)\n

# Constructor
    TransConv3d(ichannels::Int, ochannels::Int, fn::FunOrNil=relu;
                kernel   :: Dims{3} = (3,3,3),
                dilation :: Dims{3} = (1,1,1),
                stride   :: Dims{3} = (1,1,1),
                padding  :: Pads3OrStr = "valid",
                type     :: Type = Array{Float32})
+ `ichannels` is the number of input's channels
+ `ochannels` is the number of output's channels
+ `padding` can be "valid", "same", or type `NTuple{3, Dims{2}}`
"""
function TransConv3d(ichannels::Int, ochannels::Int, fn::FunOrNil=relu;
                kernel   :: Dims{3} = (3,3,3),
                dilation :: Dims{3} = (1,1,1),
                stride   :: Dims{3} = (1,1,1),
                padding  :: Pads3OrStr = "valid",
                type     :: Type = Array{Float32})

    return TransConv{3}(ichannels, ochannels, fn; kernel, dilation, stride, padding, type)
end



"""
Applies a `4`-D transpose convolution over an `6`-D input tensor of shape (ichannels, `w1`,`w2`,`w3`,`w4`, batchsize)\n

# Constructor
    TransConv4d(ichannels::Int, ochannels::Int, fn::FunOrNil=relu;
                kernel   :: Dims{4} = (3,3,3,3),
                dilation :: Dims{4} = (1,1,1,1),
                stride   :: Dims{4} = (1,1,1,1),
                padding  :: Pads4OrStr = "valid",
                type     :: Type = Array{Float32})
+ `ichannels` is the number of input's channels
+ `ochannels` is the number of output's channels
+ `padding` can be "valid", "same", or type `NTuple{4, Dims{2}}`
"""
function TransConv4d(ichannels::Int, ochannels::Int, fn::FunOrNil=relu;
                     kernel   :: Dims{4} = (3,3,3,3),
                     dilation :: Dims{4} = (1,1,1,1),
                     stride   :: Dims{4} = (1,1,1,1),
                     padding  :: Pads4OrStr = "valid",
                     type     :: Type = Array{Float32})

    return TransConv{4}(ichannels, ochannels, fn; kernel, dilation, stride, padding, type)
end


"""
Applies a `5`-D transpose convolution over an `7`-D input tensor of shape (ichannels, `w1`,`w2`,`w3`,`w4`,`w5`, batchsize)\n

# Constructor
    TransConv5d(ichannels::Int, ochannels::Int, fn::FunOrNil=relu;
                kernel   :: Dims{5} = (3,3,3,3,3),
                dilation :: Dims{5} = (1,1,1,1,1),
                stride   :: Dims{5} = (1,1,1,1,1),
                padding  :: Pads5OrStr = "valid",
                type     :: Type = Array{Float32})
+ `ichannels` is the number of input's channels
+ `ochannels` is the number of output's channels
+ `padding` can be "valid", "same", or type `NTuple{5, Dims{2}}`
"""
function TransConv5d(ichannels::Int, ochannels::Int, fn::FunOrNil=relu;
                     kernel   :: Dims{5} = (3,3,3,3,3),
                     dilation :: Dims{5} = (1,1,1,1,1),
                     stride   :: Dims{5} = (1,1,1,1,1),
                     padding  :: Pads5OrStr = "valid",
                     type     :: Type = Array{Float32})

    return TransConv{5}(ichannels, ochannels, fn; kernel, dilation, stride, padding, type)
end
