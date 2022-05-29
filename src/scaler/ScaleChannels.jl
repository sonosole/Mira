"""
# Summary ScaleChannels
    mutable struct ScaleChannels <: Scaler
# Fields
    scale::VarOrNil
    views::Tuple
Applies scalar multiplication over channel dimension of a N-dimensional input.
This scale vector is learnable.
# Example
If the `input` has size (C,H,W,B), then you should use :

`ScaleChannels(xxx; ndims=4, keptdim=1, keptsize=C)` and size(`scale`)==(C,1,1,1)

"""
mutable struct ScaleChannels <: Scaler
    scale::VarOrNil
    views::Tuple
    function ScaleChannels(scalar::AbstractFloat;
                           ndims::Int,
                           keptdim::Int,
                           keptsize::Int,
                           type::Type=Array{Float32})
        @assert ndims > 0 "ndims > 0, but got ndims=$ndims"
        @assert ndims >= keptdim >= 1 "keptdim in [1, $ndims], but got keptdim=$keptdim"

        shape = ntuple(i -> i==keptdim ? keptsize : 1, ndims);
        views = ntuple(i -> i>=keptdim ? i+1 : i, ndims-1);

        scale = Variable{type}(Zeros(type, shape) .+ eltype(type)(scalar), true, true, true);
        new(scale, views)
    end
    function ScaleChannels(views::Tuple)
        new(nothing, views)
    end
end


function clone(this::ScaleChannels; type::Type=Array{Float32})
    cloned = ScaleChannels(this.views)
    cloned.scale = clone(this.scale, type=type)
    return cloned
end

function Base.show(io::IO, m::ScaleChannels)
    print(io, "ScaleChannels(size(scale)==$(size(m.scale)); type=$(typeof(m.scale.value)))")
end

function paramsof(m::ScaleChannels)
    params = Vector{Variable}(undef,1)
    params[1] = m.scale
    return params
end

function xparamsof(m::ScaleChannels)
    xparams = Vector{XVariable}(undef,1)
    xparams[1] = ('w', m.scale)
    return xparams
end

function nparamsof(m::ScaleChannels)
    return length(m.scale)
end

function bytesof(m::ScaleChannels, unit::String="MB")
    return blocksize(sizeof(m.scale), uppercase(unit))
end


function forward(m::ScaleChannels, x::Variable{T}) where T
    k = m.scale
    y = Variable{T}(ᵛ(x) .* ᵛ(k), x.backprop)

    if y.backprop
        y.backward = function ∇ScaleChannels()
            if need2computeδ!(x) δ(x) .+=     δ(y) .* ᵛ(k)                end
            if need2computeδ!(k) δ(k) .+= sum(δ(y) .* ᵛ(x), dims=m.views) end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
        addchild(y, k)
    end
    return y
end


function predict(m::ScaleChannels, x::AbstractArray)
    k = m.scale.value
    return x .* k
end
