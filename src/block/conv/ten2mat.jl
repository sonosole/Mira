export ten2mat
const IntStepRanges{D} = NTuple{D, StepRange{Int64, Int64}} where D
const IntUnitRange     = UnitRange{Int64}

struct Ten2matFwdIter{D}
    stride     :: NTuple{D, Int64}  # stride of filter kernels
    zwidth     :: NTuple{D, Int64}  # spatial widths of Conv's output
    istart     :: IntStepRanges{D}  # first patch's range at each dims
    ibatchsize :: UnitRange{Int64}  # all samples indices in a batch, i.e. 1:batchsize
    ichannel   :: UnitRange{Int64}  # input channel indices, i.e. 1:channel
    irows      :: UnitRange{Int64}  # rows indices in the so-called im2col algorithm
    cols       :: Int64             # number of columns of the output in im2col algorithm
    npatches   :: Int64             # total patches of Conv's input
    function Ten2matFwdIter(ekernel   :: Dims{D},   # equivalent kernel sizes when dilation>1
                            dilation  :: Dims{D},   # dilation of filter kernels
                            stride    :: Dims{D},   # stride of filter kernels
                            zwidth    :: Dims{D},   # spatial widths of Conv's output
                            rows      :: Int,       # rows of the output in im2col algorithm
                            cols      :: Int,       # cols of the output in im2col algorithm
                            npatches  :: Int,       # total patches of Conv's input
                            nchannels :: Int,       # number of input channels
                            batchsize :: Int) where D

        InitPatchRange = ntuple(i -> 1 : dilation[i] : ekernel[i], D)
        SamplesRange   = 1 : batchsize
        ChannelRange   = 1 : nchannels
        RowsRange      = 1 : rows
        new{D}(stride, zwidth, InitPatchRange, SamplesRange, ChannelRange, RowsRange, cols, npatches)
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
    coords = batched_patch_coords(n, T.zwidth)
    # teniters is like (1:nchannels, range1, range2, ..., rangeD, 1:batchsize)
    teniters = ntuple(D+2) do i
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
    matiters = (T.irows, n : T.npatches : T.cols)
    return CartesianIndices(matiters), CartesianIndices(teniters)
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


"""
    ten2matFwdInfo(x        :: AbstractArray,
                   padding  :: Pads{D},
                   kernel   :: Dims{D},
                   dilation :: Dims{D},
                   stride   :: Dims{D}) -> ROWS, COLS, batchsize, FwdIter

As the name suggestes, it returns the forward informations needed when convert tensor `x` into matrix.
+ `x` is Conv's input before padding
+ `ROWS` is the number of matrix rows when `x` is converted into matrix
+ `COLS` is the number of matrix cols when `x` is converted into matrix
"""
function ten2matFwdInfo(x        :: AbstractArray,
                        padding  :: Pads{D},
                        kernel   :: Dims{D},
                        dilation :: Dims{D},
                        stride   :: Dims{D}) where D

    assertdim(x, D+2)
    sizeofx = size(x)
    xwidth  = ntuple(i -> sizeofx[i+1] + sum(padding[i]), D)           # equivalent input spatial width
    ekernel = ntuple(i -> dilation[i] * (kernel[i] - 1) + 1, D)        # equivalent kernel sizes
    zwidth  = ntuple(i -> (xwidth[i] - ekernel[i]) ÷ stride[i] + 1, D) # equivalent output spatial width

    xchannels = sizeofx[1]
    batchsize = sizeofx[2+D]
    npatches  = prod(zwidth)           # total moving steps along all D dims
    ROWS = xchannels * prod(kernel)    # total number of elements of a patch
    COLS = batchsize * npatches        # total moving steps in a batch
    Iter = Ten2matFwdIter(ekernel, dilation, stride, zwidth, ROWS, COLS, npatches, xchannels, batchsize)

    return ROWS, COLS, batchsize, Iter
end


"""
    ten2mat
A part of Conv module
# Explain
The nomal convolution is X = Conv(Z), decomposed into following:
```julia
   ┌──────────────────────────────────────────────────────────────────┐
   │ ┌────────────────────────┐         ┌───────────┐       ┌───────┐ │
X →│ │[padfn] → Xten ← [tomat]│→ Xmat → │ W*(∙) + B │ → Y → │reshape│ │→ Z
   │ └────────ten2mat─────────┘         └───Dense───┘       └───────┘ │
   └──────────────────────────────Conv────────────────────────────────┘
```
"""
function ten2mat(x        :: Array{T},
                 padding  :: Pads{D},
                 kernel   :: Dims{D},
                 dilation :: Dims{D},
                 stride   :: Dims{D},
                 padmode  :: Function = padconst,
                 padval   :: Real = 0) where {T,D}

    rows, cols, batchsize, FwdIter = ten2matFwdInfo(x, padding, kernel, dilation, stride)

    if padmode == padconst
        xten = padmode(x, extendpad(padding), padval)
    else
        xten = padmode(x, extendpad(padding))
    end

    xmat = similar(xten, rows, cols)

    Threads.@threads for (m, t) in FwdIter
        @inbounds xmat[m] .= reshape(xten[t], rows, batchsize)
    end

    return xmat
end


# (x) → ten2mat → (y);
# (x) → ConvND  → (z);
mutable struct Ten2matBwdIter{D}
    zrange :: NTuple{D, UnitRange{Int64}}    # spatial range of ConvND's output
    zwidth :: NTuple{D, Int64}               # spatial width of ConvND's output
    bools  :: NTuple{D, Bool}                # parallelizable dimension is true
    shape  :: Vector{Int}                    # all unparallelizable dims forms a new idx matrix
    total  :: Int                            # total elements of aforementioned idx matrix
    function Ten2matBwdIter(zwidth::Dims{D}, bools::NTuple{D, Bool}) where D
        count = 0
        total = 1
        shape = Vector{Int}(undef, D - sum(bools))
        for (i, parallelizable) in enumerate(bools)
            if !parallelizable
                count += 1
                total *= zwidth[i]
                shape[count] = zwidth[i]
            end
        end
        new{D}(ntuple(i -> 1:zwidth[i], D), zwidth, bools, shape, total)
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
            return iter.zrange[d]
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

    rows, cols, batchsize, FwdIter = ten2matFwdInfo(ᵛ(x), padding, kernel, dilation, stride)

    if padmode == padconst
        xten = padmode(x, extendpad(padding), padval)
    else
        xten = padmode(x, extendpad(padding))
    end

    mat = similar(ᵛ(xten), rows, cols)

    Threads.@threads for (m, t) in FwdIter
        @inbounds mat[m] .= reshape(xten.value[t], rows, batchsize)
    end
    xmat = Variable{Array{T}}(mat, xten.backprop)

    if xmat.backprop
        parallizable = ntuple(i -> dilation[i] * (kernel[i] - 1) + 1, D) .≤ stride

        xmat.backward = function ∇ten2mat()
            if needgrad(xten)
                zerodelta(xten)
                if !all(parallizable)
                    BwdIter = Ten2matBwdIter(FwdIter.zwidth, parallizable)
                    for pindices in BwdIter
                        # locally parallel calculation
                        Threads.@threads for coords in pindices
                            n = coords2nth(BwdIter.zwidth, coords)
                            m, t = FwdIter[n]
                            @inbounds xten.delta[t] .+= reshape(xmat.delta[m], size(xten.delta[t]))
                        end
                    end
                else
                    # globally parallel calculation
                    Threads.@threads for (m, t) in FwdIter
                        @inbounds xten.delta[t] .+= reshape(xmat.delta[m], size(xten.delta[t]))
                    end
                end
            end
            ifNotKeepδThenFreeδ!(xmat)
        end

        addchild(xmat, xten)
    end

    return xmat
end
