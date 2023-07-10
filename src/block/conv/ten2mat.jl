export ten2mat
const IntStepRanges{D} = NTuple{D, StepRange{Int64, Int64}} where D
const IntUnitRange     = UnitRange{Int64}

struct Ten2matFwdIter{D}
    stride     :: NTuple{D, Int64}  # stride of filter kernels
    zsize      :: NTuple{D, Int64}  # shape of the output feature in n-dim conv, exclude batch and channel dims
    istart     :: IntStepRanges{D}  # first patch's range at each dims
    ibatchsize :: UnitRange{Int64}  # all samples indices in a batch, i.e. 1:batchsize
    ichannel   :: UnitRange{Int64}  # input channel indices, i.e. 1:channel
    irows      :: UnitRange{Int64}  # rows indices in the so-called im2col algorithm
    cols       :: Int64             # number of columns of the output in im2col algorithm
    npatches   :: Int64             # npatches of one sample (batchsize==1)
    function Ten2matFwdIter(ekernel   :: Dims{D},   # equivalent kernel sizes when dilation>1
                            dilation  :: Dims{D},   # dilation of filter kernels
                            stride    :: Dims{D},   # stride of filter kernels
                            zsize     :: Dims{D},   # shape of the output feature in n-dim conv
                            rows      :: Int,       # rows of the output in im2col algorithm
                            cols      :: Int,       # cols of the output in im2col algorithm
                            npatches  :: Int,       # npatches of one sample (batchsize==1)
                            nchannels :: Int,       # number of input channels
                            batchsize :: Int) where D

        InitPatchRange = ntuple(i -> 1 : dilation[i] : ekernel[i], D)
        SamplesRange   = 1 : batchsize
        ChannelRange   = 1 : nchannels
        RowsRange      = 1 : rows
        new{D}(stride, zsize, InitPatchRange, SamplesRange, ChannelRange, RowsRange, cols, npatches)
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

@inline function coords2nth(dims::Dims{D}, coords::Dims{D}) where D
    nth = first(coords)
    stride = accumulate(*, dims)
    for i in 1:D-1
        nth += stride[i] * (coords[i+1] - 1)
    end
    return nth
end

@inline function coords2nth(dims::Dims{D}, coords::CartesianIndex{D}) where D
    nth = first(coords.I)
    stride = accumulate(*, dims)
    for i in 1:D-1
        nth += stride[i] * (coords.I[i+1] - 1)
    end
    return nth
end

function Base.getindex(T::Ten2matFwdIter{D}, n::Int) where D
    coords = batched_patch_coords(n, T.zsize)
    # xiters is like (1:nchannels, range1, range2, ..., rangeD, 1:batchsize)
    xiters = ntuple(D+2) do i
        if isequal(i, 1)
            return T.ichannel
        end
        if i ≤ D+1
            j = i - 1
            offset = (coords[j] - 1) * T.stride[j]
            return T.istart[j] .+ offset
        end
        return T.ibatchsize
    end
    # length(n:steps:cols)==batchsize
    yiters = (T.irows, n : T.npatches : T.cols)
    return CartesianIndices(yiters), CartesianIndices(xiters)
end


Base.length(T::Ten2matFwdIter)     = T.npatches
Base.lastindex(T::Ten2matFwdIter)  = T.npatches
Base.firstindex(T::Ten2matFwdIter) = 1

function Base.iterate(T::Ten2matFwdIter, i::Int=1)
    if i ≤ T.npatches
        return T[i], i+1
    else
        return nothing
    end
end



function ten2matFwdInfo(x        :: AbstractArray,
                        padding  :: Pads{D},
                        kernel   :: Dims{D},
                        dilation :: Dims{D},
                        stride   :: Dims{D}) where D

    assertdim(x, D+2)
    sizeofx = size(x)
    xsize   = ntuple(i -> sizeofx[i+1] + sum(padding[i]), D)           # equivalent spatial width
    ekernel = ntuple(i -> dilation[i] * (kernel[i] - 1) + 1, D)        # equivalent kernel sizes
    zsize   = ntuple(i -> (xsize[i] - ekernel[i]) ÷ stride[i] + 1, D)  # equivalent moving steps, i.e. output feature spatial width

    xchannels = sizeofx[1]
    batchsize = sizeofx[2+D]
    npatches  = prod(zsize)            # total moving steps along all D dims
    ROWS = xchannels * prod(kernel)    # total number of elements of a patch
    COLS = npatches * batchsize        # total moving steps in a batch
    Iter = Ten2matFwdIter(ekernel, dilation, stride, zsize, ROWS, COLS, npatches, xchannels, batchsize)

    return ROWS, COLS, batchsize, Iter
end


function ten2mat(x        :: Array{T},
                 padding  :: Pads{D},
                 kernel   :: Dims{D},
                 dilation :: Dims{D},
                 stride   :: Dims{D},
                 padmode  :: Function = padconst,
                 padval   :: Real = 0) where {T,D}

    rows, cols, batchsize, YXIndices = ten2matFwdInfo(x, padding, kernel, dilation, stride)

    if padmode == padconst
        x = padmode(x, extendpad(padding), padval)
    else
        x = padmode(x, extendpad(padding))
    end

    y = similar(x, rows, cols)

    Threads.@threads for (o, i) in YXIndices
        @inbounds y[o] .= reshape(x[i], rows, batchsize)
    end

    return y
end


# (x) → ten2mat → (y);
# (x) → ConvND  → (z);
mutable struct Ten2matBwdIter{D}
    zsize :: NTuple{D, UnitRange{Int64}}    # spatial range of ConvND's output
    sizez :: NTuple{D, Int64}               # spatial width of ConvND's output
    bools :: NTuple{D, Bool}                # parallelizable dimension is true
    shape :: Vector{Int}                    # all unparallelizable dims forms a new idx matrix
    total :: Int                            # total elements of aforementioned idx matrix
    function Ten2matBwdIter(zsize::Dims{D}, bools::NTuple{D, Bool}) where D
        count = 0
        total = 1
        shape = Vector{Int}(undef, D - sum(bools))
        for (i, parallelizable) in enumerate(bools)
            if !parallelizable
                count += 1
                total *= zsize[i]
                shape[count] = zsize[i]
            end
        end
        new{D}(ntuple(i -> 1:zsize[i], D), zsize, bools, shape, total)
    end
end

@inline function unparallel_patch_coords(n::Int, dims::Vector{Int})
    j = n - 1
    coords = ntuple(length(dims)) do i
        r = mod(j, dims[i]) + 1  # rows   at dims[i]
        j = div(j, dims[i])      # cols-1 at dims[i+1]
        return r
    end
    return coords
end

function Base.getindex(iter::Ten2matBwdIter{D}, n::Int) where D
    # coords that can't be processed in parallel
    coords = unparallel_patch_coords(n, iter.shape)
    counts = 0

    parallelidxs = ntuple(D) do d
        if iter.bools[d]
            return iter.zsize[d]
        else
            counts += 1
            return coords[counts] : coords[counts]
        end
    end
    return CartesianIndices(parallelidxs)
end


Base.length(T::Ten2matBwdIter)     = T.total
Base.lastindex(T::Ten2matBwdIter)  = T.total
Base.firstindex(T::Ten2matBwdIter) = 1

function Base.iterate(T::Ten2matBwdIter, i::Int=1)
    if i ≤ T.total
        return T[i], i+1
    else
        return nothing
    end
end


function ten2mat(x        :: Variable{Array{T}},
                 padding  :: Pads{D},
                 kernel   :: Dims{D},
                 dilation :: Dims{D},
                 stride   :: Dims{D},
                 padmode  :: Function = padconst,
                 padval   :: Real = 0) where {T,D}

    rows, cols, batchsize, YXIndices = ten2matFwdInfo(ᵛ(x), padding, kernel, dilation, stride)

    if padmode == padconst
        px = padmode(x, extendpad(padding), padval)
    else
        px = padmode(x, extendpad(padding))
    end

    vy = similar(ᵛ(x), rows, cols)

    Threads.@threads for (o, i) in YXIndices
        @inbounds vy[o] .= reshape(px.value[i], rows, batchsize)
    end
    y = Variable{Array{T}}(vy, px.backprop)

    if y.backprop
        parallizable = ntuple(i -> dilation[i] * (kernel[i] - 1) + 1, D) .≤ stride

        y.backward = function ∇ten2mat()
            if need2computeδ!(px)
                zerodelta(px)
                if !all(parallizable)
                    BwdIter = Ten2matBwdIter(YXIndices.zsize, parallizable)
                    for pindices in BwdIter
                        # locally parallel calculation
                        Threads.@threads for coords in pindices
                            n = coords2nth(BwdIter.sizez, coords)
                            o, i = YXIndices[n]
                            @inbounds px.delta[i] .+= reshape(y.delta[o], size(px.delta[i]))
                        end
                    end
                else
                    # globally parallel calculation
                    Threads.@threads for (o, i) in YXIndices
                        @inbounds px.delta[i] .+= reshape(y.delta[o], size(px.delta[i]))
                    end
                end
            end
            ifNotKeepδThenFreeδ!(y)
        end

        addchild(y, px)
    end

    return y
end
