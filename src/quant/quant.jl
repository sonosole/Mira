export Quant


"""
supports both per tensor and per channel asymmetric and asymmetric linear quantization.
Per tensor means that all the values within the tensor are scaled the same way.
Per channel means that channel dimension of a tensor are scaled and offset by a
different value. Note that, we ensure that zero in floating point is represented
with no error after quantization, thereby ensuring that operations like padding
do not cause additional quantization error.
"""
mutable struct Quant
    Xmin::Real
    Xmax::Real
    Qmin::Integer
    Qmax::Integer
    S::Real
    Z::Integer
    function Quant(Xmin::Real, Xmax::Real, Qmin::Integer, Qmax::Integer)
        @assert 0 ≤ Xmin ≤ Xmax "0 ≤ Xmin ≤ Xmax, but got Xmin=$Xmin Xmax=$Xmax"
        @assert 0 ≤ Qmin ≤ Qmax "0 ≤ Qmin ≤ Qmax, but got Qmin=$Qmin Qmax=$Qmax"
        scale, zeropoint = scale_and_zeropoint(Xmin, Xmax, Qmin, Qmax)
        new(Xmin, Xmax, Qmin, Qmax, scale, zeropoint)
    end
end


function Base.show(io::IO, q::Quant)
    Xmin = q.Xmin
    Xmax = q.Xmax
    Qmin = q.Qmin
    Qmax = q.Qmax
    S = q.S
    Z = q.Z
    print(io, "Quant(Xmin=$Xmin, Xmax=$Xmax, Qmin=$Qmin, Qmax=$Qmax, scale=$S, zeropoint=$Z)")
end


function quantize(Q::Quant, x::AbstractArray)
    return clamp.(x .* (1 / Q.S)) .+ Q.Z, Q.Qmin, Q.Qmax)
end


function dequantize(Q::Quant, q::AbstractArray{I}) where I <: Integer
    S, Z = Q.S, Q.Z
    return S .* (q .- Z)
end


function xqx(Q::Quant, x::AbstractArray)
    q = clamp.(x .* (1 / Q.S)) .+ Q.Z, Q.Qmin, Q.Qmax)
    return Q.S .* (q .- Q.Z)
end
