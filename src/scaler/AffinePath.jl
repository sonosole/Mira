"""
# Summary AffinePath
    mutable struct AffinePath <: Scaler
    `x -> w .* x .+ b`

# Fields
    w :: VarOrNil
    b :: VarOrNil

# Example
If the `input` has size (C,H,W,B), then you should use :

`AffinePath(xxx; ndims=4)` and size(`w`)==(1,1,1,1)

"""
mutable struct AffinePath <: Scaler
    w::VarOrNil
    b::VarOrNil
    function AffinePath(w::Real, b::Real; ndims::Int, type::Type=Array{Float32})
        @assert ndims >= 1 "ndims >= 1 shall be met, but got ndims=$ndims"
        typed = eltype(type)
        shape = ntuple(i->1, ndims)
        slope = Variable{type}(Zeros(type, shape) .+ typed(w), true, true, true);
        bias  = Variable{type}(Zeros(type, shape) .+ typed(b), true, true, true);
        new(slope, bias)
    end
    function AffinePath()
        new(nothing, nothing)
    end
end


function clone(this::AffinePath; type::Type=Array{Float32})
    cloned = AffinePath()
    cloned.w = clone(this.w, type=type)
    cloned.b = clone(this.b, type=type)
    return cloned
end

function Base.show(io::IO, m::AffinePath)
    w = Array(m.w.value)[1]
    b = Array(m.b.value)[1]
    print(io, "AffinePath(w=$w, b=$b; type=$(typeof(m.w.value)))")
end

function paramsof(m::AffinePath)
    params = Vector{Variable}(undef,2)
    params[1] = m.w
    params[2] = m.b
    return params
end

function xparamsof(m::AffinePath)
    xparams = Vector{XVariable}(undef,2)
    xparams[1] = ('w', m.w)
    xparams[2] = ('b', m.b)
    return xparams
end

function nparamsof(m::AffinePath)
    return 2
end

elsizeof(a::AffinePath) = elsizeof(a.w)

function bytesof(m::AffinePath, unit::String="MB")
    return blocksize(2*sizeof(m.w), uppercase(unit))
end

nops(AffinePath) = (0, 0, 0)

function forward(m::AffinePath, x::Variable{T}) where T
    w = m.w
    b = m.b
    y = Variable{T}(ᵛ(w) .* ᵛ(x) .+ ᵛ(b), x.backprop)

    if y.backprop
        y.backward = function ∇ScalePath()
            need2computeδ!(x) && (x ←     δ(y) .* ᵛ(w) )
            need2computeδ!(w) && (w ← sum(δ(y) .* ᵛ(x)))
            need2computeδ!(b) && (b ← sum(δ(y)        ))
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
        addchild(y, w)
        addchild(y, b)
    end
    return y
end


function predict(m::AffinePath, x::AbstractArray)
    w = ᵛ(m.w)
    b = ᵛ(m.b)
    return w .* x .+ b
end
