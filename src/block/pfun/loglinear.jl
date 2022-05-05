export LogLinear

"""
    mutable struct LogLinear <: Block

Applies a linear and log transformation to the incoming data:
y = LogLinear(x) == log.(k .* x .+ b)
"""
mutable struct LogLinear <: Block
    k::VarOrNil
    b::VarOrNil
    ϵ::AbstractFloat
    views::NTuple
    function LogLinear(slope::Real,
                       bias::Real;
                       ndims::Int,                  # how many dimentions the input data has
                       keptdims::Union{Tuple,Int},  # must be unique and sorted and positive
                       keptsize::Union{Tuple,Int},  # must be positive
                       eps::AbstractFloat=1e-9,     # stability const
                       type::Type=Array{Float32})
        shape, views = Mira.ShapeAndViews(ndims, keptdims, keptsize);
        T = eltype(type)
        ϵ = T(eps)
        k = Zeros(type, shape) .+ T(slope)
        b = Zeros(type, shape) .+ T(bias)

        new(Variable{type}(k, true, true, true),
            Variable{type}(b, true, true, true), ϵ, views)
    end
    function LogLinear(ϵ::AbstractFloat, views::NTuple)
        new(nothing, nothing, ϵ, views)
    end
end

function clone(this::LogLinear; type::Type=Array{Float32})
    cloned = LogLinear(this.ϵ, this.views)
    cloned.k = clone(this.k, type=type)
    cloned.b = clone(this.b, type=type)
    return cloned
end


# pretty show
function Base.show(io::IO, m::LogLinear)
    k = m.k.value
    b = m.b.value
    println(io, "═════════ LogLinear ═════════")
    print(io, cyan!(" slope k "))
    display(k)
    print(io, cyan!(" offset b "))
    display(b)
end


function paramsof(m::LogLinear)
    params = Vector{Variable}(undef,2)
    params[1] = m.k
    params[2] = m.b
    return params
end


function xparamsof(m::LogLinear)
    xparams = Vector{XVariable}(undef,2)
    xparams[1] = ('o', m.k)
    xparams[2] = ('o', m.b)
    return xparams
end


function nparamsof(m::LogLinear)
    return length(m.k) + length(m.b)
end


elsizeof(l::LogLinear) = elsizeof(l.k)


function bytesof(model::LogLinear, unit::String="MB")
    n = nparamsof(model) * elsizeof(model)
    return blocksize(n, uppercase(unit))
end


function nops(l::LogLinear)
    @info "ops of LogLinear depends on the length of input, so it can't be inffered"
    return (0, 0, 0) # (mul, add, act)
end


# lazy forward
# function forward(m::LogLinear, x::Variable)
#     k = m.k
#     b = m.b
#     ϵ = m.ϵ
#     return log(abs(k) .* x .+ abs(b) + ϵ)
# end

# ops fused forward
function forward(m::LogLinear, x::Variable{T}) where T
    k = m.k
    b = m.b
    ϵ = m.ϵ
    t = abs(ᵛ(k)) .* ᵛ(x) .+ abs(ᵛ(b)) .+ ϵ
    y = Variable{T}(log.(t), x.backprop)

    if y.backprop
        y.backward = function LogLinearBackward()
            z = eltype(T)(1) ./ t
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* z .* abs(ᵛ(k))
            end
            if need2computeδ!(k)
                δ(k) .+= sum(δ(y) .* z .* ᵛ(x) .* sign.(ᵛ(k)), dims=m.views)
            end
            if need2computeδ!(b)
                δ(b) .+= sum(δ(y) .* z .* sign.(ᵛ(b)), dims=m.views)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
        addchild(y, k)
        addchild(y, b)
    end
    return y
end


function predict(m::LogLinear, x)
    k = abs.(m.k.value)
    b = abs.(m.b.value) .+ m.ϵ
    return @. log(k * x + b)
end
