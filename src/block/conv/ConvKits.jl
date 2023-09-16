const Pads{D}    = NTuple{D, Dims{2}}                                where D
const PadsDOrStr = Union{Int, NTuple{D, Union{Dims{2},Int}}, String} where D
const Pads1OrStr = Union{Int,                 Dims{2},       String} # for conv1d
const Pads2OrStr = Union{Int, NTuple{2, Union{Dims{2},Int}}, String} # for conv2d
const Pads3OrStr = Union{Int, NTuple{3, Union{Dims{2},Int}}, String} # for conv3d
const Pads4OrStr = Union{Int, NTuple{4, Union{Dims{2},Int}}, String} # for conv4d
const Pads5OrStr = Union{Int, NTuple{5, Union{Dims{2},Int}}, String} # for conv5d
const DimsOrNil  = Union{Dims, Nil}

@doc """
`Pads{D}` is a very strict type for padding infomations, for a D-dims Conv,
the padding pixels at each dim is described with a Tuple{Int,Int}
""" Pads

@doc """
`PadsDOrStr` is a loose type for passing padding infomations for a N-dims Conv, it allows the following types
+ `String`, can be "same" or "valid"
+ `Int`, all dims use the same padding, e.g. ((3,3), (3,3), (3,3))
+ `NTuple{D, Int}`, each dim's left and right padding are given with the same Int-typed value, e.g. ((2,2), (7,7))
+ `NTuple{D, Dims{2}}`, each dim's left and right padding are given with Dims{2}-typed value, e.g. ((2,9), (6,5))
""" PadsDOrStr


@inline function singletuple(i::Int)
    return (i,)
end
@inline function singletuple(ij::Dims{2})
    return (ij,)
end

function selectpad(padmode::String)
    if padmode == "zeros"
        return padconst
    elseif padmode == "constant"
        return padconst
    elseif padmode == "repeat"
        return padrepeat
    elseif padmode == "reflect"
        return padreflect
    elseif padmode == "symmetric"
        return padsymmetric
    elseif padmode == "circular"
        return padcircular
    else
        error("padmode should be one of \"zeros\", \"constant\", \"repeat\", \"reflect\", \"symmetric\",\"circular\"")
    end
end


function inferpadding(padding::String, kernel::Int, stride::Int, dilation::Int)
    # valid: size(input) ≥ size(output)
    # same: size(input) == size(output)
    if padding ∉ ("same", "valid")
        error("padmode should be \"zeros\" or \"const\", but got $padding")
    end
    if isequal(padding, "same")
        if stride ≠ 1
            error("when padding==\"same\", stride should be 1, but only got stride=$stride")
        end
        leftpad  = div(dilation*(kernel-1), 2, RoundUp)
        rightpad = div(dilation*(kernel-1), 2, RoundDown)
    end
    if isequal(padding, "valid")
        leftpad  = 0
        rightpad = 0
    end
    return ((leftpad, rightpad),)
end


function inferpadding(padding::String, kernel::Dims{D}, stride::Dims{D}, dilation::Dims{D}) where D
    # valid: size(input) ≥ size(output)
    # same: size(input) == size(output)
    if padding ∉ ("same", "valid")
        error("padmode should be \"zeros\" or \"const\", but got $padding")
    end
    if isequal(padding, "same")
        if prod(stride) ≠ 1
            error("when padding==\"same\", stride should be 1, but only got stride=$stride")
        end
        npads = ntuple(D) do i
            leftpad  = div(dilation[i]*(kernel[i]-1), 2, RoundUp)
            rightpad = div(dilation[i]*(kernel[i]-1), 2, RoundDown)
            return (leftpad, rightpad)
        end
    end
    if isequal(padding, "valid")
        npads = ntuple(i -> (0,0), D)
    end
    return npads
end


# spatialpadding -> (channelspadding, spatialpadding, batchsizepadding)
"""
    extendpad(padding::Pads{D}) where D
Extend spatial-padding by `C`hannels-padding and `B`atchsize-padding
# Example
            Hight Paddings      Width Paddings
                         ↓      ↓
    julia> extendpad( ((1,2), (2,3)) )
    ((0, 0), (1, 2), (2, 3), (0, 0))
       ↑       ↑      ↑        ↑
       C       H      W        B
"""
@inline function extendpad(padding::Pads{D}) where D
    return ntuple(i -> (1 < i < D+2) ? padding[i-1] : (0,0), D+2)
end


"""
Return full size of conv layer's output
"""
function fullsize(w        :: AbstractArray, # weights of conv layer
                  x        :: AbstractArray, # input before padding
                  padding  :: Pads{D},
                  kernel   :: Dims{D},
                  dilation :: Dims{D},
                  stride   :: Dims{D}) where D

    sizeofx = size(x)
    xsize   = ntuple(i -> sizeofx[i+1] + sum(padding[i]), D)    # equivalent spatial width after padding
    ekernel = ntuple(i -> dilation[i] * (kernel[i] - 1) + 1, D) # equivalent kernel size when dialating

    N = 2 + D
    zchannels = size(w, 1)
    batchsize = sizeofx[N]

    return ntuple(N) do j
        if isequal(j, 1)
            return zchannels
        end
        if 1 < j < N
            i = j - 1
            return (xsize[i] - ekernel[i]) ÷ stride[i] + 1
        end
        return batchsize
    end
end

"""
Return full size of conv layer's output
"""
function fullsize(w        :: Variable, # weights of conv layer
                  x        :: Variable, # input before padding
                  padding  :: Pads{D},
                  kernel   :: Dims{D},
                  dilation :: Dims{D},
                  stride   :: Dims{D}) where D

    sizeofx = size(x)
    xsize   = ntuple(i -> sizeofx[i+1] + sum(padding[i]), D)    # equivalent spatial width after padding
    ekernel = ntuple(i -> dilation[i] * (kernel[i] - 1) + 1, D) # equivalent kernel size when dialating

    N = 2 + D
    zchannels = size(w, 1)
    batchsize = sizeofx[N]

    return ntuple(N) do j
        if isequal(j, 1)
            return zchannels
        end
        if 1 < j < N
            i = j - 1
            return (xsize[i] - ekernel[i]) ÷ stride[i] + 1
        end
        return batchsize
    end
end



"""
Return full size of pooling layer's output
"""
function poolsize(x        :: AbstractArray, # input before padding
                  padding  :: Pads{D},
                  kernel   :: Dims{D},
                  dilation :: Dims{D},
                  stride   :: Dims{D}) where D

    sizeofx = size(x)
    xsize   = ntuple(i -> sizeofx[i+1] + sum(padding[i]), D)    # equivalent spatial width after padding
    ekernel = ntuple(i -> dilation[i] * (kernel[i] - 1) + 1, D) # equivalent kernel size when dialating

    N = 2 + D
    xchannels = size(x, 1)
    batchsize = sizeofx[N]

    return ntuple(N) do j
        if isequal(j, 1)
            return xchannels
        end
        if 1 < j < N
            i = j - 1
            return (xsize[i] - ekernel[i]) ÷ stride[i] + 1
        end
        return batchsize
    end
end

"""
Return full size of pooling layer's output
"""
function poolsize(x        :: Variable, # input before padding
                  padding  :: Pads{D},
                  kernel   :: Dims{D},
                  dilation :: Dims{D},
                  stride   :: Dims{D}) where D

    sizeofx = size(x)
    xsize   = ntuple(i -> sizeofx[i+1] + sum(padding[i]), D)    # equivalent spatial width after padding
    ekernel = ntuple(i -> dilation[i] * (kernel[i] - 1) + 1, D) # equivalent kernel size when dialating

    N = 2 + D
    xchannels = size(x, 1)
    batchsize = sizeofx[N]

    return ntuple(N) do j
        if isequal(j, 1)
            return xchannels
        end
        if 1 < j < N
            i = j - 1
            return (xsize[i] - ekernel[i]) ÷ stride[i] + 1
        end
        return batchsize
    end
end
