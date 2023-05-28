"""
# Summary ScalePath
    mutable struct ScalePath <: Scaler
Scale a N-dimensional input by learnable `w` parameter.
# Fields
    w::VarOrNil
# Example
If the `input` has size (C,H,W,B), then you should use :

`ScalePath(xxx; ndims=4)` and size(`w`)==(1,1,1,1)

"""
mutable struct ScalePath <: Scaler
    w::VarOrNil
    function ScalePath(scalar::AbstractFloat; ndims::Int, type::Type=Array{Float32})
        @assert ndims >= 1 "ndims >= 1 shall be met, but got ndims=$ndims"
        shape = ntuple(i->1, ndims)
        scale = Variable{type}(Zeros(type, shape) .+ eltype(type)(scalar), true, true, true);
        new(scale)
    end
    function ScalePath()
        new(nothing)
    end
end


function clone(this::ScalePath; type::Type=Array{Float32})
    cloned = ScalePath()
    cloned.w = clone(this.w, type=type)
    return cloned
end

function Base.show(io::IO, m::ScalePath)
    print(io, "ScalePath($(m.w.value[1]); type=$(typeof(m.w.value)))")
end

function paramsof(m::ScalePath)
    params = Vector{Variable}(undef,1)
    params[1] = m.w
    return params
end

function xparamsof(m::ScalePath)
    xparams = Vector{XVariable}(undef,1)
    xparams[1] = ('w', m.w)
    return xparams
end

function nparamsof(m::ScalePath)
    return 1
end

function bytesof(m::ScalePath, unit::String="MB")
    return blocksize(sizeof(m.w), uppercase(unit))
end


function forward(m::ScalePath, x::Variable{T}) where T
    w = m.w
    y = Variable{T}(ᵛ(x) .* ᵛ(w), x.backprop)

    if y.backprop
        y.backward = function ∇ScalePath()
            if need2computeδ!(x)
                x ← δ(y) .* ᵛ(w)
            end
            if need2computeδ!(w)
                w ← sum(δ(y) .* ᵛ(x)) .+ zero(w)
            end
            ifNotKeepδThenFreeδ!(y);
        end
        addchild(y, x)
        addchild(y, w)
    end
    return y
end


function predict(m::ScalePath, x::AbstractArray)
    w = ᵛ(m.w)
    return w .* x
end
