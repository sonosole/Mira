export scale
export zeropoint
export scale_and_zeropoint
export xqx, xqx!, quantize, dequantize


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
    S = scale(Xmin, Xmax, Qmin, Qmax) # scale
    Z = Qmax - round(I, Xmax / S)     # zero-point
    return Z
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
    quantize(x::Real, Xmin::Real, Xmax::Real, Qmin::Integer, Qmax::Integer) -> xq::dtype

where [Xmin, Xmax] denotes the range of the input data while Qmin and Qmax are
respectively the minimum and maximum values of the quantized data type.
```
julia> quantize(0.5, 0.0, 1.0, 0, 5)
2
```
"""
function quantize(x::Real, Xmin::Real, Xmax::Real, Qmin::I, Qmax::I) where I <: Integer
    S, Z = scale_and_zeropoint(Xmin, Xmax, Qmin, Qmax)
    x = clamp(x, Xmin, Xmax)
    q = round(I, x / S) + Z
    return q
end


function quantize(x::AbstractArray, Xmin::Real, Xmax::Real, Qmin::I, Qmax::I) where I <: Integer
    S, Z = scale_and_zeropoint(Xmin, Xmax, Qmin, Qmax)
    K = 1 / S
    x = @. clamp(x, Xmin, Xmax)
    q = @. round(I, x * K) + Z
    return q
end


"""
    dequantize(x::Real, Xmin::Real, Xmax::Real, Qmin::Integer, Qmax::Integer; dtype=Float32) -> xq::dtype

where [Xmin, Xmax] denotes the range of the input data while Qmin and Qmax are
respectively the minimum and maximum values of the quantized data type.
```
julia> dequantize(0.5, 0.0, 1.0, 0, 5)
0.4
```
"""
function dequantize(q::I, Xmin::Real, Xmax::Real, Qmin::I, Qmax::I; dtype=Float32) where I <: Integer
    S, Z = scale_and_zeropoint(Xmin, Xmax, Qmin, Qmax)
    return type(S * (q - Z))
end


function dequantize(q::AbstractArray{I}, Xmin::Real, Xmax::Real, Qmin::I, Qmax::I; type=Array{Float32}) where I <: Integer
    S, Z = scale_and_zeropoint(Xmin, Xmax, Qmin, Qmax)
    x = @. S * (q - Z)
    return type(x)
end


"""
    xqx(X::Real, Xmin::Real, Xmax::Real, Qmin::Integer, Qmax::Integer) -> Ẋ::typeof(X)

    Ẋ = dequantize( quantize( X ) )
```
julia> xqx(0.5, 0.0, 1.0, 0, 5)
0.4
```
"""
function xqx(x::Real, Xmin::Real, Xmax::Real, Qmin::I, Qmax::I) where I <: Integer
    S, Z = scale_and_zeropoint(Xmin, Xmax, Qmin, Qmax)
    K = 1 / S
    T = typeof(x)
    x = clamp(x, Xmin, Xmax)
    q = round(I, x * K) + Z
    return T(S * (q - Z))
end

function xqx(x::AbstractArray, Xmin::Real, Xmax::Real, Qmin::I, Qmax::I) where I <: Integer
    S, Z = scale_and_zeropoint(Xmin, Xmax, Qmin, Qmax)
    K = 1 / S
    T = typeof(x)
    x = @. clamp(x, Xmin, Xmax)
    q = @. round(I, x * K) + Z
    return T(S .* (q .- Z))
end

function xqx!(x::AbstractArray, Xmin::Real, Xmax::Real, Qmin::I, Qmax::I) where I <: Integer
    S, Z = scale_and_zeropoint(Xmin, Xmax, Qmin, Qmax)
    K  = 1 / S
    x  = clamp!(x, Xmin, Xmax)
    q  = round.(I, x .* K) .+ Z
    x .= S .* (q .- Z)
    return x
end
