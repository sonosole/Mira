"""
    min2max(k .* x .+ 0.5)
"""
mutable struct HardSigmoid
    k     :: VarOrNil # scaling params
    views :: NTuple   # views to collect elements
    function HardSigmoid(slope::Real;
                         ndims::Int,
                         keptdims::Union{Tuple,Int}, # must be unique and sorted and positive
                         keptsize::Union{Tuple,Int}, # must be positive
                         type::Type=Array{Float32})

        shape, views = ShapeAndViews(ndims, keptdims, keptsize);
        k = Variable{type}(Zeros(type, shape) .+ eltype(type)(slope), true, true, true);
        new(k, views)
    end
    function HardSigmoid(views::NTuple)
        new(nothing, views)
    end
end


function clone(this::HardSigmoid; type::Type=Array{Float32})
    cloned = HardSigmoid(this.views)
    cloned.k = clone(this.k, type=type)
    return cloned
end


function Base.show(io::IO, h::HardSigmoid)
    TYPE = typeof(h.k.value)
    print(io, "HardSigmoid(type=$TYPE)")
    display(io, h.k.value)
end


function paramsof(h::HardSigmoid)
    params = Vector{Variable}(undef,1)
    params[1] = h.k
    return params
end

function xparamsof(h::HardSigmoid)
    xparams = Vector{XVariable}(undef,1)
    xparams[1] = ('o', h.k)
    return xparams
end

function nparamsof(h::HardSigmoid)
    return length(h.k)
end

elsizeof(h::HardSigmoid) = elsizeof(h.k)

function bytesof(h::HardSigmoid, unit::String="MB")
    n = nparamsof(h) * elsizeof(h)
    return blocksize(n, uppercase(unit))
end

function forward(h::HardSigmoid, x::Variable{T}) where T
    k = h.k
    y = min2max(k .* x + 0.5, lower=0.0, upper=1.0)
    return y
end


function predict(h::HardSigmoid, x::AbstractArray)
    k = h.k.value
    b = eltype(x)(0.5)
    return min2max(k .* x .+ b, lower=0.0, upper=1.0)
end
