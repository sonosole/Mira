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
        @assert Xmin ≤ Xmax "Xmin ≤ Xmax, but got Xmin=$Xmin Xmax=$Xmax"
        @assert Qmin ≤ Qmax "Qmin ≤ Qmax, but got Qmin=$Qmin Qmax=$Qmax"
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
    S = Q.S
    Z = Q.Z
    K = 1/S
    x = clamp.(x, Q.Xmin, Q.Xmax)
    q = round.(eltype(Z), x .* K) .+ Z
    return q
end


function dequantize(Q::Quant, q::AbstractArray{I}; type=Array{Float32}) where I <: Integer
    S, Z = Q.S, Q.Z
    return type(S .* (q .- Z))
end


function xqx(Q::Quant, x::AbstractArray)
    S = Q.S
    Z = Q.Z
    K = 1/S
    T = typeof(x)
    x = clamp.(x, Q.Xmin, Q.Xmax)
    q = round.(eltype(Z), x .* K) .+ Z
    return T(S .* (q .- Z))
end


function xqx!(Q::Quant, x::AbstractArray)
    S = Q.S
    Z = Q.Z
    K = 1/S
    x  = clamp!(x, Q.Xmin, Q.Xmax)
    q  = round.(eltype(Z), x .* K) .+ Z
    x .= S .* (q .- Z)
    return x
end
