export decay
export LpNormClip
export L2NormClip
export L1NormClip
export L0NormClip
export LPInfNormClip
export LNInfNormClip
export setNanInfZero, setNanInfZero!
export clip!

"""
    clip!(::Vector{XVariable}, kind='u'; L1decay=0.0, L2decay=0.0, clipvalue=1.0)

Limit the amplitude of parameters. `kind` has four options:\n
`'u'` for recurrent params\n
`'b'` for bias params\n
`'w'` for projection params\n
`'a'` for `'u'`, `'b'` and `'w'` params\n
as show in `yᵗ = f(w*xᵗ + u*hᵗ⁻¹ + b)` or other similar formulas
"""
function clip!(xparams::Vector{XVariable}, kind='u'; L1decay=0.0, L2decay=0.0, clipvalue=1.0)
    @assert clipvalue>0 "clipvalue is positive, but got $clipvalue"
    if !(kind=='u' || kind=='b' || kind=='w' || kind=='a')
        @error "type of XVariable not among u/w/b/a, but got $kind"
    end

    λ₁ = -L1decay
    λ₂ = -L2decay
    Threads.@threads for (c, θ) in xparams
        if c == kind || kind=='a'
            𝒗 = ᵛ(θ)
            i = abs.(𝒗) .> clipvalue
            if sum(i) == 0 continue end
            if λ₁==0 && λ₂==0                     # Hard truncation
                @. 𝒗[i] = clipvalue * sign(𝒗[i])
            elseif λ₁==0 && λ₂!=0                 # Soft truncation (L2)
                @. 𝒗[i] += λ₂ * 𝒗[i]
            elseif λ₁!=0 && λ₂==0                 # Gradual truncation (L1)
                @. 𝒗[i] += λ₁ * sign(𝒗[i])
            else  # λ₁!=0 && λ₂!=0
                @. 𝒗[i] += λ₁ * sign(𝒗[i]) + λ₂ * 𝒗[i]
            end
        end
    end
end


function decay(params::Vector{Variable}; ratio=0.999)
    for p in params
        p.value .*= ratio
    end
end


"""
    setNanInfZero(x)
```julia
x = randn(1,4)
x[1] = Inf;
x[2] =-Inf;
x[3] = NaN;
```
```
julia> x
1×4 Array{Float64,2}:
 Inf  -Inf  NaN  0.602655

julia> x = setNanInfZero(x)
1×4 Array{Float64,2}:
 0.0  0.0  0.0  0.602655
```
 """
function setNanInfZero(x)
    x[ isnan.(x) .⊻ isinf.(x) ] .= 0.0
    return x
end


function setNanInfZero!(x)
    x[ isnan.(x) .⊻ isinf.(x) ] .= 0.0
    return nothing
end


function L2NormClip(x::AbstractArray, clipvalue)
    pnorm = sqrt(sum(x.^2) / length(x))
    scale = clipvalue / pnorm
    if pnorm > clipvalue
        x .*= scale
    end
    return x
end


function L1NormClip(x::AbstractArray, clipvalue)
    pnorm = sum(abs.(x)) / length(x)
    scale = clipvalue / pnorm
    if pnorm > clipvalue
        x .*= scale
    end
    return x
end


function L0NormClip(x::AbstractArray, clipvalue)
    pnorm = sum(x .!= 0.0) / length(x)
    scale = clipvalue / pnorm
    if pnorm > clipvalue
        x .*= scale
    end
    return x
end


function LPInfNormClip(x::AbstractArray, clipvalue)
    pnorm = maximum(abs.(x))
    scale = clipvalue / pnorm
    if pnorm > clipvalue
        x .*= scale
    end
    return x
end


function LNInfNormClip(x::AbstractArray, clipvalue)
    pnorm = minimum(abs.(x))
    scale = clipvalue / pnorm
    if pnorm > clipvalue
        x .*= scale
    end
    return x
end


function LpNormClip(x::AbstractArray, clipvalue; order::Union{Int,String}=2)
    order==2 && return L2NormClip(x, clipvalue)
    order==1 && return L1NormClip(x, clipvalue)
    order==0 && return L0NormClip(x, clipvalue)
    order=="inf"  && return LPInfNormClip(x, clipvalue)
    order=="-inf" && return LNInfNormClip(x, clipvalue)
    pnorm = (sum( abs.(x).^order ) / length(x)) ^ (1/order)
    scale = clipvalue / pnorm
    if pnorm > clipvalue
        x .*= scale
    end
    return x
end
