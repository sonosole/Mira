export im2col

"""
    row_col_index(n::Int, rows::Int) -> (row, col)
Return the row and col index of the n-th element of a matrix having shape (`rows`, *)
"""
@inline function row_col_index(n::Int, rows::Int)
    n = n - 1
    r = mod(n, rows) + 1  # row index
    c = div(n, rows) + 1  # col index
    return (r, c)
end

"""
    row_col_batch_index(n::Int, rows::Int, cols::Int) -> (row, col, batch)
Return the row, col and batch index of the n-th element of a array having shape (`rows`, `col`, *)
"""
@inline function row_col_batch_index(n::Int, rows::Int, cols::Int)
    r, j = row_col_index(n, rows)
    c, b = row_col_index(j, cols)
    return (r, c, b)
end

"""
    patchcoords(n::Int, dims::NTuple{D,Int}) where D

Return the `n`-th element's coords in array `x` having shape of (W1,W2,W3,...,WD, Batchsize).
`dims` = (W1,W2,W3,...,WD).
"""
@inline function patchcoords(n::Int, dims::NTuple{D,Int}) where D
    coords = Vector{Int}(undef, D+1)
    i = 1   # iter index for `Idxs`
    j = n   # inital total elements
    d = D   # iter index for `dims` argument
    while d > 0
        coords[i], j = row_col_index(j, dims[i])
        i += 1
        d -= 1
    end
    coords[D+1] = j

    return coords
end

@inline function patchcoords(n::Int, dims::Vector{Int})
    D = length(dims)
    coords = Vector{Int}(undef, D+1)
    i = 1   # iter index for `Idxs`
    j = n   # inital total elements
    d = D   # iter index for `dims` argument
    while d > 0
        coords[i], j = row_col_index(j, dims[i])
        i += 1
        d -= 1
    end
    coords[D+1] = j

    return coords
end



struct Im2colFwdIter{D}
    ekernel   :: NTuple{D,Int}   # equivalent kernel sizes for D dims
    dilation  :: NTuple{D,Int}   # dilation of filter kernels
    stride    :: NTuple{D,Int}   # stride of filter kernels
    zsize     :: NTuple{D,Int}   # shape of the output feature in ConvND, batch and channel dims excluded
    rows      :: Int             # rows of the output in im2col algorithm
    cols      :: Int             # cols of the output in im2col algorithm
    channels  :: Int             # number of input channels
    function Im2colFwdIter(ekernel  :: NTuple{D,Int},
                           dilation :: NTuple{D,Int},
                           stride   :: NTuple{D,Int},
                           zsize    :: NTuple{D,Int},
                           rows     :: Int,
                           cols     :: Int,
                           channels :: Int) where D
        new{D}(ekernel, dilation, stride, zsize, rows, cols, channels)
    end
end

function Base.show(io::IO, ::MIME"text/plain", I::Im2colFwdIter{D}) where D
    println("Im2colFwdIter{$D}")
    println("  ", I.ekernel,  "\t → equivalent kernel size")
    println("  ", I.dilation, "\t → dilation")
    println("  ", I.stride,   "\t → stride")
    println("  ", I.zsize,    "\t → size of output features")
    println("  ", I.rows,     "\t → rows in im2col")
    println("  ", I.cols,     "\t → cols in im2col")
    println("  ", I.channels, "\t → input channels")
end

function Base.show(io::IO, ::MIME"text/plain", I::Im2colFwdIter{1})
    println("Im2colFwdIter{1}")
    println("  ", I.ekernel[1],  "\t → equivalent kernel size")
    println("  ", I.dilation[1], "\t → dilation")
    println("  ", I.stride[1],   "\t → stride")
    println("  ", I.zsize[1],    "\t → size of output features")
    println("  ", I.rows,        "\t → rows in im2col")
    println("  ", I.cols,        "\t → cols in im2col")
    println("  ", I.channels,    "\t → input channels")
end


function Base.getindex(I::Im2colFwdIter{D}, n::Int) where D
    coords = patchcoords(n, I.zsize)
    xiters = ntuple(D+2) do i
        if isequal(i, 1)
            return 1 : I.channels
        end
        if i ≤ D+1
            j = i - 1
            istart = 1 : I.dilation[j] : I.ekernel[j]
            offset = (coords[j] - 1) * I.stride[j]
            return istart .+ offset
        end
        return coords[D+1] : coords[D+1]
    end
    yiters = (1 : I.rows, n : n)
    return CartesianIndices(yiters), CartesianIndices(xiters)
end


Base.length(I::Im2colFwdIter)     = I.cols
Base.lastindex(I::Im2colFwdIter)  = I.cols
Base.firstindex(I::Im2colFwdIter) = 1

function Base.iterate(I::Im2colFwdIter, i::Int=1)
    if i ≤ I.cols
        return I[i], i+1
    else
        return nothing
    end
end


function im2colFwdInfo(x        :: AbstractArray,
                       padding  :: NTuple{D,NTuple{2,Int}},
                       kernel   :: NTuple{D,Int},
                       dilation :: NTuple{D,Int},
                       stride   :: NTuple{D,Int}) where D

    assertdim(x, D+2)
    sizeofx = size(x)
    xsize   = ntuple(i -> sizeofx[i+1] + sum(padding[i]), D)             # equivalent spatial width
    ekernel = ntuple(i -> dilation[i] * (kernel[i] - 1) + 1, D)          # equivalent kernel sizes
    zsize   = ntuple(i -> (xsize[i] - ekernel[i]) ÷ stride[i] + 1, D)    # equivalent moving steps, i.e. output feature spatial width

    xchannels = sizeofx[1]
    batchsize = sizeofx[2+D]
    ROWS = xchannels * prod(kernel)    # nelements of a patch
    COLS = batchsize * prod(zsize)     # total moving steps in a batch
    Iter = Im2colFwdIter(ekernel, dilation, stride, zsize, ROWS, COLS, xchannels)

    return ROWS, COLS, Iter
end



function im2col(x        :: Array{T},
                padding  :: NTuple{D,NTuple{2,Int}},
                kernel   :: NTuple{D,Int},
                dilation :: NTuple{D,Int},
                stride   :: NTuple{D,Int},
                padval   :: Real = 0) where {T,D}

    rows, cols, YXIndices = im2colFwdInfo(x, padding, kernel, dilation, stride)

    x = padconst(x, extendpad(padding), padval)
    y = similar(x, rows, cols)

    Threads.@threads for (o, i) in YXIndices
        @inbounds y[o] .= reshape(x[i], rows, 1)
    end

    return y
end


# (x) → im2col → (y);
# (x) → ConvND → (z);
mutable struct Im2colBwdIter{N}
    zsizeb :: NTuple{N, UnitRange{Int64}}    # spatial range of ConvND's output concat batchsize
    sizezb :: NTuple{N, Int64}               # spatial width of ConvND's output concat batchsize
    boolsb :: NTuple{N, Bool}                # parallelizable dimension is true
    shape  :: Vector{Int}                    # all unparallelizable dims forms a new idx matrix
    total  :: Int                            # total elements of aforementioned idx matrix
    function Im2colBwdIter(zsize::NTuple{D, Int64}, bools::NTuple{D, Bool}, batchsize::Int) where D
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
        zsizeb = ntuple(i -> i ≤ D ? (1:zsize[i]) : (1:batchsize), D+1)
        sizezb = ntuple(i -> i ≤ D ? zsize[i] : batchsize, D+1)
        boolsb = ntuple(i -> i ≤ D ? bools[i] : true, D+1)
        new{D+1}(zsizeb, sizezb, boolsb, shape, total)
    end
end

function Base.getindex(iter::Im2colBwdIter{N}, n::Int) where N
    counts = 0
    coords = unparallel_patch_coords(n, iter.shape)
    parallelidxs = ntuple(N) do i
        if iter.boolsb[i]
            return iter.zsizeb[i]
        else
            counts += 1
            return coords[counts] : coords[counts]
        end
    end
    return CartesianIndices(parallelidxs)
end


Base.length(I::Im2colBwdIter)     = I.total
Base.lastindex(I::Im2colBwdIter)  = I.total
Base.firstindex(I::Im2colBwdIter) = 1

function Base.iterate(I::Im2colBwdIter, i::Int=1)
    if i ≤ I.total
        return I[i], i+1
    else
        return nothing
    end
end


function im2col(x        :: Variable{Array{T}},
                padding  :: NTuple{D,Dims{2}},
                kernel   :: NTuple{D,Int},
                dilation :: NTuple{D,Int},
                stride   :: NTuple{D,Int},
                padval   :: Real = 0) where {T,D}

    rows, cols, YXIndices = im2colFwdInfo(ᵛ(x), padding, kernel, dilation, stride)

    px = padconst(x, extendpad(padding), padval)
    vy = similar(ᵛ(x), rows, cols)

    Threads.@threads for (o, i) in YXIndices
        @inbounds vy[o] .= reshape(px.value[i], rows, 1)
    end
    y = Variable{Array{T}}(vy, px.backprop)

    if y.backprop
        parallizable = YXIndices.ekernel .≤ stride

        y.backward = function ∇im2col()
            if need2computeδ!(px)
                zerodelta(px)
                if !all(parallizable)
                    BwdIter = Im2colBwdIter(YXIndices.zsize, parallizable, size(x, ndims(x)))
                    for pindices in BwdIter
                        # locally parallel calculation
                        Threads.@threads for coords in pindices
                            n = coords2nth(BwdIter.sizezb, coords)
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
