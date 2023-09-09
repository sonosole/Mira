export Pool
export Pool1d
export Pool2d
export Pool3d
export Pool4d
export Pool5d


"""
Applies a `D`-dim pooling over an `(D+2)`-dim input tensor of shape (ichannels, w1, w2, ..., wn, batchsize)\n
# Constructor
    Pool{D}(poolingf :: Function;
            kernel   :: Dims{D} = ntuple(i -> 2, D),
            dilation :: Dims{D} = ntuple(i -> 1, D),
            stride   :: Dims{D} = ntuple(i -> 2, D),
            padval   :: Real = 0f0,
            padmode  :: String  = "zeros",
            padding  :: PadsDOrStr = "valid") where D
+ `poolingf` can be user-defined, e.g. `softmax`, but usally `maximum`(aka MaxPool) or "mean"(aka AvgPool)
+ `padmode` should be one of \"zeros\", \"constant\", \"repeat\", \"reflect\", \"symmetric\", \"circular\"
+ `padding` can be \"valid\", \"same\", or type `NTuple{D, Dims{2}}`
"""
mutable struct Pool{D} <: Block
    f        :: FunOrNil
    kernel   :: Dims{D}
    dilation :: Dims{D}
    stride   :: Dims{D}
    padding  :: Pads{D}
    padmode  :: Function
    padval   :: Float32
    function Pool{D}(poolingf :: Function;
                     kernel   :: Dims{D} = ntuple(i -> 2, D),
                     dilation :: Dims{D} = ntuple(i -> 1, D),
                     stride   :: Dims{D} = ntuple(i -> 2, D),
                     padval   :: Real = 0f0,
                     padmode  :: String  = "zeros",
                     padding  :: PadsDOrStr = "valid") where D

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
        if isequal(poolingf, max)
            padval = -Inf
        end
        new{D}(poolingf, kernel, dilation, stride,
               npads, selectpad(padmode), padval)
    end
    function Pool{D}() where D
        O = (0, 0)
        new{D}(nothing,
               ntuple(i -> 2, D),
               ntuple(i -> 1, D),
               ntuple(i -> 2, D),
               ntuple(i -> O, D), padconst, 0f0)
    end
end


function clone(this::Pool{D}; type::Type=Array{Float32}) where D
    cloned   = Pool{D}()
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
function Base.show(io::IO, p::Pool{1})
    P = ifelse(paddings(p.padding)==0, "", " padding=$(first(p.padding)),")
    D = ifelse(first(p.dilation)==1,   "", " dilation=$(first(p.dilation)),")
    S = ifelse(first(p.stride)==1,     "", " stride=$(first(p.stride)),")

    DSP = "$D$S$P"
    FUN = p.f

    if FUN == maximum
        print(io, "MaxPool1d(kernel=$(first(p.kernel)),$DSP $FUN)")
    elseif FUN == mean
        print(io, "AvgPool1d(kernel=$(first(p.kernel)),$DSP $FUN)")
    else
        F = uppercasefirst(string(p.f))
        print(io, "Pool1d(kernel=$(first(p.kernel)),$DSP $FUN)")
    end
end

function Base.show(io::IO, p::Pool{N}) where N
    P = ifelse(paddings(p.padding)==0, "", " padding=$(p.padding),")
    D = ifelse(prod(p.dilation)==1,    "", " dilation=$(p.dilation),")
    S = ifelse(prod(p.stride)==1,      "", " stride=$(p.stride),")

    DSP = "$D$S$P"
    FUN = p.f

    if FUN == maximum
        print(io, "MaxPool$(N)d(kernel=$(p.kernel),$DSP $FUN)")
    elseif FUN == mean
        print(io, "AvgPool$(N)d(kernel=$(p.kernel),$DSP $FUN)")
    else
        F = uppercasefirst(string(p.f))
        print(io, "$(F)Pool$(N)d(kernel=$(p.kernel),$DSP $FUN)")
    end
end


function paramsof(c::Pool{D}) where D
    return nothing
end


function xparamsof(c::Pool{D}) where D
    return nothing
end


function nparamsof(c::Pool{D}) where D
    return 0
end


function bytesof(c::Pool{D}, unit::String="MB") where D
    return 0
end



function forward(C::Pool{D}, x::Variable) where D
    S = poolsize(x, C.padding, C.kernel, C.dilation, C.stride)
    y =  ten2mat(x, C.padding, C.kernel, C.dilation, C.stride, C.padmode, C.padval)
    xchannels = size(x, 1)
    kernellen = prod(C.kernel)
    columns   = size(y, 2)
    z = reshape(y, xchannels,kernellen,columns)
    return reshape(C.f(z, dims=2), S)
end


function predict(C::Pool{D}, x::AbstractArray) where D
    S = poolsize(x, C.padding, C.kernel, C.dilation, C.stride)
    y =  ten2mat(x, C.padding, C.kernel, C.dilation, C.stride, C.padmode, C.padval)
    xchannels = size(x, 1)
    kernellen = prod(C.kernel)
    columns   = size(y, 2)
    z = reshape(y, xchannels,kernellen,columns)
    return reshape(C.f(z, dims=2), S)
end



"""
Applies a `1`-D pooling over an `3`-D input tensor of shape (ichannels, `steps`, batchsize)\n

# Constructor
    Pool1d(poolingf :: Function;
           kernel   :: Int = 2,
           dilation :: Int = 1,
           stride   :: Int = 2,
           padval   :: Real = 0f0,
           padmode  :: String = "repeat",
           padding  :: Dims2OrStr = "valid")
+ `poolingf` can be user-defined, e.g. `softmax`, but usally `maximum`(aka MaxPool) or "mean"(aka AvgPool)
+ `padmode` should be one of \"zeros\", \"constant\", \"repeat\", \"reflect\", \"symmetric\", \"circular\"
+ `padding` can be \"valid\", \"same\", or type `Dims{2}`
"""
function Pool1d(poolingf :: Function;
                kernel   :: Int = 2,
                dilation :: Int = 1,
                stride   :: Int = 2,
                padval   :: Real = 0f0,
                padmode  :: String = "repeat",
                padding  :: Pads1OrStr = "valid")

    if !isa(padding, String)
        padding = singletuple(padding)
    end

    kernel   = singletuple(kernel)
    dilation = singletuple(dilation)
    stride   = singletuple(stride)

    return Pool{1}(poolingf;
                   kernel, dilation, stride,
                   padval, padmode, padding)
end


"""
Applies a `2`-D pooling over an `4`-D input tensor of shape (ichannels, `hight`, `width`, batchsize)\n

# Constructor
    Pool2d(poolingf :: Function;
           kernel   :: Dims{2} = (2,2),
           dilation :: Dims{2} = (1,1),
           stride   :: Dims{2} = (2,2),
           padval   :: Real = 0f0,
           padmode  :: String = "repeat",
           padding  :: Pads2OrStr = "valid")
+ `poolingf` can be user-defined, e.g. `softmax`, but usally `maximum`(aka MaxPool) or "mean"(aka AvgPool)
+ `padmode` should be one of \"zeros\", \"constant\", \"repeat\", \"reflect\", \"symmetric\", \"circular\"
+ `padding` can be \"valid\", \"same\", or type `NTuple{2, Dims{2}}`
"""
function Pool2d(poolingf :: Function;
                kernel   :: Dims{2} = (2,2),
                dilation :: Dims{2} = (1,1),
                stride   :: Dims{2} = (2,2),
                padval   :: Real = 0f0,
                padmode  :: String = "repeat",
                padding  :: Pads2OrStr = "valid")

    return Pool{2}(poolingf;
                   kernel, dilation, stride,
                   padval, padmode, padding)
end



"""
Applies a `3`-D pooling over an `5`-D input tensor of shape (ichannels, `hight`, `width`, `steps`, batchsize)\n

# Constructor
    Pool3d(poolingf :: Function;
           kernel   :: Dims{3} = (2,2,2),
           dilation :: Dims{3} = (1,1,1),
           stride   :: Dims{3} = (2,2,2),
           padval   :: Real = 0f0,
           padmode  :: String = "repeat",
           padding  :: Pads3OrStr = "valid")
+ `poolingf` can be user-defined, e.g. `softmax`, but usally `maximum`(aka MaxPool) or "mean"(aka AvgPool)
+ `padmode` should be one of \"zeros\", \"constant\", \"repeat\", \"reflect\", \"symmetric\", \"circular\"
+ `padding` can be \"valid\", \"same\", or type `NTuple{3, Dims{2}}`
"""
function Pool3d(poolingf :: Function;
                kernel   :: Dims{3} = (2,2,2),
                dilation :: Dims{3} = (1,1,1),
                stride   :: Dims{3} = (2,2,2),
                padval   :: Real = 0f0,
                padmode  :: String = "repeat",
                padding  :: Pads3OrStr = "valid")

    return Pool{3}(poolingf;
                   kernel, dilation, stride,
                   padval, padmode, padding)
end



"""
Applies a `4`-D pooling over an `6`-D input tensor of shape (ichannels, `w1`,`w2`,`w3`,`w4`, batchsize)\n

# Constructor
    Pool4d(poolingf :: Function;
           kernel   :: Dims{4} = (2,2,2,2),
           dilation :: Dims{4} = (1,1,1,1),
           stride   :: Dims{4} = (2,2,2,2),
           padval   :: Real = 0f0,
           padmode  :: String = "repeat",
           padding  :: Pads4OrStr = "valid")
+ `poolingf` can be user-defined, e.g. `softmax`, but usally `maximum`(aka MaxPool) or "mean"(aka AvgPool)
+ `padmode` should be one of \"zeros\", \"constant\", \"repeat\", \"reflect\", \"symmetric\", \"circular\"
+ `padding` can be \"valid\", \"same\", or type `NTuple{4, Dims{2}}`
"""
function Pool4d(poolingf :: Function;
                kernel   :: Dims{4} = (2,2,2,2),
                dilation :: Dims{4} = (1,1,1,1),
                stride   :: Dims{4} = (2,2,2,2),
                padval   :: Real = 0f0,
                padmode  :: String = "repeat",
                padding  :: Pads4OrStr = "valid")

    return Pool{4}(poolingf;
                   kernel, dilation, stride,
                   padval, padmode, padding)
end


"""
Applies a `5`-D pooling over an `7`-D input tensor of shape (ichannels, `w1`,`w2`,`w3`,`w4`,`w5`, batchsize)\n

# Constructor
    Pool5d(poolingf :: Function;
           kernel   :: Dims{5} = (2,2,2,2,2),
           dilation :: Dims{5} = (1,1,1,1,1),
           stride   :: Dims{5} = (2,2,2,2,2),
           padval   :: Real = 0f0,
           padmode  :: String = "repeat",
           padding  :: Pads5OrStr = "valid")
+ `poolingf` can be user-defined, e.g. `softmax`, but usally `maximum`(aka MaxPool) or "mean"(aka AvgPool)
+ `padmode` should be one of \"zeros\", \"constant\", \"repeat\", \"reflect\", \"symmetric\", \"circular\"
+ `padding` can be \"valid\", \"same\", or type `NTuple{5, Dims{2}}`
"""
function Pool5d(poolingf :: Function;
                kernel   :: Dims{5} = (2,2,2,2,2),
                dilation :: Dims{5} = (1,1,1,1,1),
                stride   :: Dims{5} = (2,2,2,2,2),
                padval   :: Real = 0f0,
                padmode  :: String = "repeat",
                padding  :: Pads5OrStr = "valid")

    return Pool{5}(poolingf;
                   kernel, dilation, stride,
                   padval, padmode, padding)
end
