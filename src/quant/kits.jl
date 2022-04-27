export scale
export zeropoint
export scale_and_zeropoint
export xqx, quantize, dequantize


"""
    scale(Xmin::Real, Xmax::Real, Qmin::Integer, Qmax::Integer) -> (Xmax - Xmin) / (Qmax - Qmin)

where [Xmin, Xmax] denotes the range of the input data while Qmin and Qmax are
respectively the minimum and maximum values of the quantized data type.
```
julia> scale(0, 1.0,  0, 10)
0.1
```
"""
function scale(Xmin::Real, Xmax::Real, Qmin::I, Qmax::I) where I <: Integer
    return (Xmax - Xmin) / (Qmax - Qmin)
end

"""
    zeropoint(Xmin::Real, Xmax::Real, Qmin::Integer, Qmax::Integer)
where [Xmin, Xmax] denotes the range of the input data while Qmin and Qmax are
respectively the minimum and maximum values of the quantized data type.

```
julia> zeropoint(0,1.0, 0,10)
0

julia> zeropoint(0,1.0, -127,127)
-127
```
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
```
julia> scale_and_zeropoint(0,127.0, -127,127)
(0.5, -127)
```
"""
function scale_and_zeropoint(Xmin::Real, Xmax::Real, Qmin::I, Qmax::I) where I <: Integer
    S = (Xmax - Xmin) / (Qmax - Qmin) # scale
    Z =  Qmax - round(I, Xmax / S)    # zero-point
    return S,  Z
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
    x = clamp(x, Xmin, Xmax)
    q = round(I, x * K) + Z
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
    x = clamp(x, Xmin, Xmax)
    q = round(I, x * K + Z)
    return S * (q - Z)
end
