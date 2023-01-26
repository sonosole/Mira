"""
    IIRMeanNorm, i.e. ⤦\n
    hᵗ = w*hᵗ⁻¹ + (1-w)*xᵗ
    yᵗ = xᵗ - hᵗ
"""
mutable struct IIRMeanNorm <: Block
    w::VarOrNil # smoother weight
    h::Any      # hidden variable
    function IIRMeanNorm(scalar::Real;
                         ndims::Int,
                         keptdims::Union{Tuple,Int},
                         keptsize::Union{Tuple,Int},
                         type::Type=Array{Float32})

        array = [i for i in keptsize]
        shape = ntuple(i -> i in keptdims ? popfirst!(array) : 1, ndims)
        T = eltype(type)
        w = Variable{type}(zeros(type, shape) .+ T(scalar), true, true, true);
        new(w, nothing)
    end
    function IIRMeanNorm()
        new(nothing, nothing)
    end
end


function clone(this::IIRMeanNorm; type::Type=Array{Float32})
    cloned = IIRMeanNorm()
    cloned.w = clone(this.w, type=type)
    return cloned
end


function Base.show(io::IO, m::IIRMeanNorm)
    SIZE = size(m.w)
    TYPE = typeof(m.w.value)
    print(io, "IIRMeanNorm(size=$SIZE; type=$TYPE)")
end


function resethidden(m::IIRMeanNorm)
    m.h = nothing
end


function forward(m::IIRMeanNorm, x::Variable{T}) where T
    w = m.w
    γ = 1 - ᵛ(w)
    h = m.h ≠ nothing ? m.h : Variable(Zeros(T, size(x)), type=T)
    m.h = w .* h + γ .* x
    return x - m.h
end


function predict(m::IIRMeanNorm, x::T) where T
    w = ᵛ(m.w)
    γ = eltype(T)(1) .- w
    h = m.h ≠ nothing ? m.h : Zeros(T, size(x))
    m.h = w .* h + γ .* x
    return x - m.h
end


function paramsof(m::IIRMeanNorm)
    params = Vector{Variable}(undef,1)
    params[1] = m.w
    return params
end


function xparamsof(m::IIRMeanNorm)
    xparams = Vector{XVariable}(undef,1)
    xparams[1] = ('w', m.w)
    return xparams
end


function nparamsof(m::IIRMeanNorm)
    return length(m.w)
end

elsizeof(block::IIRMeanNorm) = elsizeof(block.w)

function bytesof(block::IIRMeanNorm, unit::String="MB")
    n = nparamsof(block) * elsizeof(block)
    return blocksize(n, uppercase(unit))
end


function nops(rnn::IIRMeanNorm, c::Int=1)
    return (0, 0, 0)
end
