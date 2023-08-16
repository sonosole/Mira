export storage

const KiB = 1024
const MiB = 1048576
const GiB = 1073741824
const TiB = 1099511627776
const PiB = 1125899906842624
const EiB = 1152921504606846976
const ZiB = 1180591620717411303424
const YiB = 1208925819614629174706176
const BiB = 1237940039285380274899124224


function storage(x, u::String="MB")
    bytes = length(x) * sizeof(eltype(x))
    return blocksize(bytes, uppercase(u))
end


function blocksize(n::Real, u::String)
    if u == "B"  return n / 1   end
    if u == "KB" return n / KiB end
    if u == "MB" return n / MiB end
    if u == "GB" return n / GiB end
    if u == "TB" return n / TiB end
    if u == "PB" return n / PiB end
    if u == "EB" return n / EiB end
    if u == "ZB" return n / ZiB end
    if u == "YB" return n / YiB end
    if u == "BB" return n / BiB end
    @error "MUST BE one of B KB MB GB TB PB EB ZB YB BB"
end


"""
    deleteat!(atuple::Tuple, iters::Union{Tuple,Array,Vector,Int}) -> Tuple
delete elements from a tuple at specified positions `iters` and return it
# Examples
    julia> deleteat!((0,2,4,6,8), (1,3,5))
    (2, 6)
"""
function Base.deleteat!(atuple::Tuple, iters::Union{Tuple,Array,Vector,Int})
    this = []; append!(this, atuple);
    if typeof(iters) <: Int
        rest = deleteat!(this, iters)
    else
        keys = []; append!(keys,iters);
        rest = deleteat!(this, keys);
    end
    return ntuple(i -> rest[i], length(rest))
end


"""
    shape_and_dims(ndims::Int,
                   keptdims::Union{Tuple,Int},
                   keptsize::Union{Tuple,Int}) -> (shape::NTuple, views::NTuple)

mainly serves for batchnorm like operations. `ndims` is the dims of input Tensor x.
`keptdims` is the dims that will be kept after reduction like mean(x,dims=`views`) and
`keptsize` is the number of elements on the `keptdims`. i.e. ⤦\n
`shape` = size( reductionFunction(x, dims=`views`) )
NOTE: If no dims to kept, then `keptdims` shall be 0
This fn is mainly used by `BatchNorm`
# Example
    julia> shape_and_dims(4, (1,4), (5,3))
       keptsize
      ↓        ↓
     (5, 1, 1, 3), (2, 3)
      1  2  3  4    ↑  ↑
      ↑        ↑    dims to reduce
       keptdims
"""
function shape_and_dims(ndims::Int,                        # ndims of input Tensor
                        keptdims::IntOrDims{D},            # must be unique and sorted and positive
                        keptsize::IntOrDims{D}) where D    # must be positive

    @assert typeof(keptsize)==typeof(keptdims) "keptsize & keptdims shall be the same type"
    @assert ndims ≥ maximum(keptdims) "ndims ≥ maximum(keptdims) shall be met"
    @assert ndims > length(keptdims) "there is no elements for statistical analysis"
    @assert ndims > 0 "ndims > 0, but got ndims=$ndims"

    if keptdims isa Int
        if keptdims == 0
            if keptsize ≠ 1
                @warn "keptsize should be 1 here, but got $keptsize"
            end
            shape = ntuple(i -> 1, ndims);
            views = ntuple(i -> i, ndims);
        else
            shape = ntuple(i -> i==keptdims ? keptsize : 1, ndims);
            views = ntuple(i -> i>=keptdims ? i+1 : i, ndims-1);
        end
    else
        array = [i for i in keptsize]
        shape = ntuple(i -> i in keptdims ? popfirst!(array) : 1, ndims);
        views = deleteat!(ntuple(i -> i, ndims), keptdims)
    end
    return shape, views
end


"""
    dimsfilter(sz::Dims{N}, dims::IntOrDims{D}) where {N,D} -> shape::Dims{N}

Make a new shape::Dims{N}, whose elements are specified by `dims` from `sz`, while
the other elements are all 1. Now used in `xdropout` and `xdropout!`

# Example
    julia> Mira.dimsfilter((20,2,3), 1)
    (20, 1, 1)
    julia> Mira.dimsfilter((7,716,522,1990,1997), (3,2))
    (1, 716, 522, 1, 1)
"""
@inline function dimsfilter(sz::Dims{N}, dims::IntOrDims{D}) where {N,D}
    shape = ones(Int, N)
    for (i, d) in enumerate(dims)
        shape[d] = sz[d]
    end
    return ntuple(i -> shape[i],N)
end


@inline function dotmul!(x::AbstractArray, y::AbstractArray)
    x .*= y
    return x
end


export isnormal
"""
    isnormal(x::Real; min::Real=-1e3, max::Real=1e3)::Bool

return true if x isn't NaN nor Inf and not exceeding range [min, max]
"""
function isnormal(x::Real; min::Real=-1e38, max::Real=1e38)
    if isnan(x) ||
       isinf(x) ||
       x > max  ||
       x < min
        return false
    else
        return true
    end
end


export cpuvec
"""
    cpuvec(x::AbstractArray) = vec(Array(x))
"""
cpuvec(x::AbstractArray) = vec(Array(x))


export ⊙
"""
    ⊙(x::BitArray, y::BitArray) = @. ! xor(x, y)
"""
function ⊙(x::BitArray, y::BitArray)
    return @. ! xor(x, y)
end


export sor
"""
    sor(x::BitArray, y::BitArray) = @. ! xor(x, y)
"""
function sor(x::BitArray, y::BitArray)
    return @. ! xor(x, y)
end
