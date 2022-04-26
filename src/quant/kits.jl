export scale
export zeropoint
export scale_and_zeropoint
export xqx, quantize, dequantize


"""
    scale(Xmin::Real, Xmax::Real, Qmin::Integer, Qmax::Integer) -> (Xmax - Xmin) / (Qmax - Qmin)

where [Xmin, Xmax] denotes the range of the input data while Qmin and Qmax are
respectively the minimum and maximum values of the quantized data type.
"""
function scale(Xmin::Real, Xmax::Real, Qmin::I, Qmax::I) where I <: Integer
    return (Xmax - Xmin) / (Qmax - Qmin)
end

"""
    zeropoint(Xmin::Real, Xmax::Real, Qmin::Integer, Qmax::Integer)
where [Xmin, Xmax] denotes the range of the input data while Qmin and Qmax are
respectively the minimum and maximum values of the quantized data type.
"""
function zeropoint(Xmin::Real, Xmax::Real, Qmin::I, Qmax::I) where I <: Integer
    S = scale(Xmin, Xmax, Qmin, Qmax)
    Z = Qmax - Xmax / S
    return round(I, Z)
end


"""
    scale_and_zeropoint(Xmin::Real, Xmax::Real, Qmin::Integer, Qmax::Integer)
where [Xmin, Xmax] denotes the range of the input data while Qmin and Qmax are
respectively the minimum and maximum values of the quantized data type.
"""
function scale_and_zeropoint(Xmin::Real, Xmax::Real, Qmin::I, Qmax::I) where I <: Integer
    S = (Xmax - Xmin) / (Qmax - Qmin)
    Z = Qmax - round(I, Xmax / S)
    return S, Z
end


"""
    quantize(x::Real, Xmin::Real, Xmax::Real, Qmin::Integer, Qmax::Integer) -> xq::Real

where [Xmin, Xmax] denotes the range of the input data while Qmin and Qmax are
respectively the minimum and maximum values of the quantized data type.

julia> quantize(0.5, 0.0, 1.0, 0, 5)
2
"""
function quantize(x::Real, Xmin::Real, Xmax::Real, Qmin::I, Qmax::I) where I <: Integer
    S, Z = scale_and_zeropoint(Xmin, Xmax, Qmin, Qmax)
    K = 1 / S
    q = clamp(x * K + Z, Qmin, Qmax)
    return q
end


"""
    dequantize(x::Real, Xmin::Real, Xmax::Real, Qmin::Integer, Qmax::Integer) -> xq::Real

where [Xmin, Xmax] denotes the range of the input data while Qmin and Qmax are
respectively the minimum and maximum values of the quantized data type.

julia> dequantize(0.5, 0.0, 1.0, 0, 5)
0.4
"""
function dequantize(q::I, Xmin::Real, Xmax::Real, Qmin::I, Qmax::I) where I <: Integer
    S, Z = scale_and_zeropoint(Xmin, Xmax, Qmin, Qmax)
    return S * (q - Z)
end


"""
    xqx(X::Real, Xmin::Real, Xmax::Real, Qmin::Integer, Qmax::Integer) -> Xq

    Xq = DeQuantize( Quantize( X ) )

julia> xqx(0.5, 0.0, 1.0, 0, 5)
0.4
"""
function xqx(x::Real, Xmin::Real, Xmax::Real, Qmin::I, Qmax::I) where I <: Integer
    S, Z = scale_and_zeropoint(Xmin, Xmax, Qmin, Qmax)
    K = 1 / S
    q = clamp(x * K + Z, Qmin, Qmax)
    return S * (q - Z)
end
