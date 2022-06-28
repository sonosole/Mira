function blocksize(n::Real, u::String)
    if u == "B"  return n / 1 end
    if u == "KB" return n / 1024 end
    if u == "MB" return n / 1048576 end
    if u == "GB" return n / 1073741824 end
    if u == "TB" return n / 1099511627776 end
    if u == "PB" return n / 1125899906842624 end
    if u == "EB" return n / 1152921504606846976 end
    if u == "ZB" return n / 1180591620717411303424 end
    if u == "YB" return n / 1208925819614629174706176 end
    if u == "BB" return n / 1237940039285380274899124224 end
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
    ShapeAndViews(ndims::Int,
                  keptdims::Union{Tuple,Int},
                  keptsize::Union{Tuple,Int}) -> (shape, views)

mainly serves for batchnorm like operations. `ndims` is the dims of input Tensor x.
`keptdims` is the dims that will be kept after reduction like mean(x,dims=`views`) and
`keptsize` is the number of elements on the `keptdims`. i.e. ⤦\n
`shape` = size( reductionFunction(x, dims=`views`) )

# Example
    julia> Delta.ShapeAndViews(4, (1,4), (5,3))
    ((5, 1, 1, 3), (2, 3))
"""
function ShapeAndViews(ndims::Int,                    # ndims of input Tensor
                       keptdims::Union{Tuple,Int},    # must be unique and sorted and positive
                       keptsize::Union{Tuple,Int})    # must be positive

    @assert typeof(keptsize)==typeof(keptdims) "keptsize & keptdims shall be the same type"
    @assert ndims >= maximum(keptdims) "ndims >= maximum(keptdims) shall be met"
    @assert ndims > length(keptdims) "this is no elements for statistical analysis"
    @assert ndims > 0 "ndims > 0, but got ndims=$ndims"

    if typeof(keptdims) <: Int
        if keptdims == 0
            if keptsize!=1
                @warn "keptsize should be 1 here, but got $keptsize"
            end
            shape = ntuple(i -> i==keptdims ? keptsize : 1, ndims);
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


@inline function dotmul!(x::AbstractArray, y::AbstractArray)
    x .*= y
    return x
end


export isnormal
function isnormal(x::Real; min::Real=-1e3, max::Real=1e3)
    if isnan(x) ||
       isinf(x) ||
       x ≥ max  ||
       x ≤ min
        return false
    else
        return true
    end
end

export cpuvec
cpuvec(x::AbstractArray) = vec(Array(x))
