const Pads{D}    = NTuple{D, NTuple{2,Int}} where D
const Dims2OrStr = Union{Dims{2}, String} # for conv1d
const Pads2OrStr = Union{Pads{2}, String} # for conv2d
const Pads3OrStr = Union{Pads{3}, String} # for conv3d
const Pads4OrStr = Union{Pads{4}, String} # for conv4d
const Pads5OrStr = Union{Pads{5}, String} # for conv5d


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
    if isequal(padding, "same") && stride≠1
        error("when padding==\"same\", stride should be 1, but only got stride=$stride")
    end
    if isequal(padding, "same") && dilation≠1
        error("when padding==\"same\", dilation should be 1, but only got dilation=$dilation")
    end

    if isequal(padding, "same")
        leftpad  = div(kernel-1, 2, RoundUp)
        rightpad = div(kernel-1, 2, RoundDown)
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
    if isequal(padding, "same") && prod(stride)≠1
        error("when padding==\"same\", stride should be 1, but only got stride=$stride")
    end
    if isequal(padding, "same") && prod(dilation)≠1
        error("when padding==\"same\", dilation should be 1, but only got dilation=$dilation")
    end

    if isequal(padding, "same")
        npads = ntuple(D) do i
            leftpad  = div(kernel[i]-1, 2, RoundUp)
            rightpad = div(kernel[i]-1, 2, RoundDown)
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
