const IntStepRanges{D} = NTuple{D, StepRange{Int64, Int64}} where D
const IntUnitRange     = UnitRange{Int64}

struct ConvFwdIter{D}
    stride     :: NTuple{D, Int}    # stride of filter kernels
    dims       :: NTuple{D, Int}    # shape of the output feature in n-dim conv
    istart     :: IntStepRanges{D}  # first patch's range at each dims
    ibatchsize :: UnitRange{Int64}  # all samples indices in a batch, i.e. 1:batchsize
    ichannel   :: UnitRange{Int64}  # input channel indices, i.e. 1:channel
    irows      :: UnitRange{Int64}  # rows indices in the so-called im2col algorithm
    cols       :: Int64             # number of columns of the output in im2col algorithm
    step       :: Int64             # npatches of one sample (batchsize==1)
    function ConvFwdIter(kernel    :: Dims{D},   # equivalent kernel sizes when dilation>1
                         dilation  :: Dims{D},   # dilation of filter kernels
                         stride    :: Dims{D},   # stride of filter kernels
                         dims      :: Dims{D},   # shape of the output feature in n-dim conv
                         rows      :: Int,       # rows of the output in im2col algorithm
                         cols      :: Int,       # cols of the output in im2col algorithm
                         step      :: Int,       # npatches of one sample (batchsize==1)
                         nchannels :: Int,       # number of input channels
                         batchsize :: Int) where D

        InitPatchRange = ntuple(i -> 1 : dilation[i] : kernel[i], D)
        SamplesRange   = 1 : batchsize
        ChannelRange   = 1 : nchannels
        RowsRange      = 1 : rows
        new{D}(stride, dims, InitPatchRange, SamplesRange, ChannelRange, RowsRange, cols, step)
    end
end

@inline function batched_patch_coords(n::Int, dims::Dims{D}) where D
    j = n - 1
    coords = ntuple(D) do i
        r = mod(j, dims[i]) + 1 # rows   at dims[i]
        j = div(j, dims[i])     # cols-1 at dims[i+1]
        return r
    end
    return coords
end


function Base.getindex(C::ConvFwdIter{D}, n::Int) where D
    coords = batched_patch_coords(n, C.dims)
    # xiters is like (1:nchannels, range1, range2, ..., rangeD, 1:batchsize)
    xiters = ntuple(D+2) do i
        if isequal(i, 1)
            return C.ichannel
        end
        if i ≤ D+1
            j = i - 1
            offset = (coords[j] - 1) * C.stride[j]
            return C.istart[j] .+ offset
        end
        return C.ibatchsize
    end
    # length(n:steps:cols)==batchsize
    yiters = (C.irows, n : C.step : C.cols)
    return CartesianIndices(yiters), CartesianIndices(xiters)
end

Base.length(C::ConvFwdIter)     = C.step
Base.lastindex(C::ConvFwdIter)  = C.step
Base.firstindex(C::ConvFwdIter) = 1

function Base.iterate(C::ConvFwdIter, i::Int=1)
    if i ≤ C.step
        return C[i], i+1
    else
        return nothing
    end
end



function ten2mat_nd_infos(x        :: AbstractArray,
                          padding  :: NTuple{D,Dims{2}},
                          kernel   :: Dims{D},
                          dilation :: Dims{D},
                          stride   :: Dims{D}) where D

    assertdim(x, D + 2)
    sizeofx   = size(x)
    CHANNELS  = sizeofx[1]
    BATCHSIZE = sizeofx[ndims(x)]

    W = ntuple(i -> sizeofx[i+1] + sum(padding[i]), D)        # equivalent spatial width
    K = ntuple(i -> dilation[i] * (kernel[i] - 1) + 1, D)   # equivalent kernel sizes
    S = ntuple(i -> (W[i] - K[i]) ÷ stride[i] + 1, D)       # equivalent moving steps, i.e. output feature size

    STEPS  = prod(S)         # total moving steps along all D dims
    KERNEL = prod(kernel)    # total effective kernel size  along all D dims

    ROWS = CHANNELS * KERNEL    # total number of elements of a patch
    COLS = STEPS * BATCHSIZE    # total moving steps in a batch
    Iter = ConvFwdIter(K, dilation, stride, S, ROWS, COLS, STEPS, CHANNELS, BATCHSIZE)

    return ROWS, COLS, BATCHSIZE, Iter
end


function tensor2matrix(x        :: Array{T},
                       padding  :: NTuple{D,NTuple{2,Int}},
                       kernel   :: NTuple{D,Int},
                       dilation :: NTuple{D,Int},
                       stride   :: NTuple{D,Int},
                       padval   :: Real = 0) where {T,D}

    rows, cols, batchsize, YXIndices = ten2mat_nd_infos(x, padding, kernel, dilation, stride)

    x = padconst(x, ntuple(i -> (1 < i < D+2) ? padding[i-1] : (0,0), D+2), padval)
    y = similar(x, rows, cols)

    Threads.@threads for (o, i) in YXIndices
        @inbounds y[o] .= reshape(x[i], rows, batchsize)
    end

    return y
end
