"""
# Summary WeightedChannels
    mutable struct WeightedChannels <: Scaler
# Fields
    w :: VarOrNil
    f :: FunOrNil

Applies channel-wise weighting: `x -> f(w) .* x` , where `x` is a N-dimensional\n
input, `w` is the learable weight for channel dimension and `f` is a differentiable\n
function define in this pkg or defined by user.

# Example
If the `input` has size (C,H,W,B), then you should use :
`WeightedChannels(xxx; ndims=4, keptdim=1, keptsize=C)` and size(`w`)==(C,1,1,1)
"""
mutable struct WeightedChannels
    w :: VarOrNil
    f :: FunOrNil
    function WeightedChannels(scalar::AbstractFloat,
                              func::FunOrNil=sigmoid;
                              ndims::Int,
                              keptdim::Int,
                              keptsize::Int,
                              type::Type=Array{Float32})
        @assert ndims > 0 "ndims > 0, but got ndims=$ndims"
        @assert ndims >= keptdim >= 1 "keptdim in [1, $ndims], but got keptdim=$keptdim"
        typed = eltype(type)
        shape = ntuple(i -> i==keptdim ? keptsize : 1, ndims);
        scale = Variable{type}(randn(typed, shape) .* typed(scalar), true, true, true);
        new(scale, func)
    end
    function WeightedChannels()
        new(nothing, nothing)
    end
end


function clone(this::WeightedChannels; type::Type=Array{Float32})
    cloned = WeightedChannels()
    cloned.w = clone(this.w, type=type)
    cloned.f = this.f
    return cloned
end

function Base.show(io::IO, m::WeightedChannels)
    len = length(m.w)
    fun = m.f
    print(io, "WeightedChannels($len, $fun)")
end

function paramsof(m::WeightedChannels)
    params = Vector{Variable}(undef,1)
    params[1] = m.w
    return params
end

function xparamsof(m::WeightedChannels)
    xparams = Vector{XVariable}(undef,1)
    xparams[1] = ('w', m.w)
    return xparams
end

function nparamsof(m::WeightedChannels)
    return length(m.w)
end

function bytesof(m::WeightedChannels, unit::String="MB")
    return blocksize(sizeof(m.w), uppercase(unit))
end

nops(::WeightedChannels) = (0, 0, 0)

function forward(m::WeightedChannels, x::Variable{T}) where T
    f = m.f
    w = m.w
    return f(w) .* x
end


function predict(m::WeightedChannels, x::AbstractArray)
    f = m.f
    w = m.w.value
    return f(w) .* x
end
