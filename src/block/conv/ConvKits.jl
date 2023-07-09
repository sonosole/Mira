const Pads{D}    = NTuple{D, NTuple{2,Int}} where D
const Dims2OrStr = Union{Dims{2}, String} # for conv1d
const Pads2OrStr = Union{Pads{2}, String} # for conv2d
const Pads3OrStr = Union{Pads{3}, String} # for conv3d
const Pads4OrStr = Union{Pads{4}, String} # for conv4d
const Pads5OrStr = Union{Pads{5}, String} # for conv5d


function selectpadfn(padmode::String)
    if padmode == "zeros"
        return padzeros
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


# padding -> (channelspadding, padding, batchsizepadding)
@inline function extendpad(padding::Pads{D}) where D
    return ntuple(i -> (1 < i < D+2) ? padding[i-1] : (0,0), D+2)
end


function spatialdims(z::AbstractArray, x::AbstractArray, k::Dims{D}, d::Dims{D}, s::Dims{D}) where D
    w = size(x)
    return ntuple(D+2) do j
        if j == 1
            return size(z, 1)
        end
        if j > 1
            i = j - 1
            return (w[j] - d[i] * (k[i] - 1) - 1) ÷ s[i] + 1
        end
    end
end
