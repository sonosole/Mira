mutable struct IndGRU <: Block
    # update gate
    Wz::VarOrNil
    Uz::VarOrNil
    bz::VarOrNil
    # reset gate
    Wr::VarOrNil
    Ur::VarOrNil
    br::VarOrNil
    # candidate
    Wc::VarOrNil
    Uc::VarOrNil
    bc::VarOrNil
    h  # hidden variable
    function IndGRU(isize::Int, hsize::Int; type::Type=Array{Float32})
        T  = eltype(type)

        Wz = randn(T, hsize, isize) .* sqrt( T(2/isize) )
        Uz = uniform(T, (hsize, 1), from=-0.2, to=0.2)
        bz = zeros(T, hsize, 1)

        Wr = randn(T, hsize, isize) .* sqrt( T(2/isize) )
        Ur = uniform(T, (hsize, 1), from=-0.2, to=0.2)
        br = zeros(T, hsize, 1)

        Wc = randn(T, hsize, isize) .* sqrt( T(2/isize) )
        Uc = uniform(T, (hsize, 1), from=-0.2, to=0.2)
        bc = zeros(T, hsize, 1)


        new(Variable{type}(Wz,true,true,true), Variable{type}(Uz,true,true,true), Variable{type}(bz,true,true,true),
            Variable{type}(Wr,true,true,true), Variable{type}(Ur,true,true,true), Variable{type}(br,true,true,true),
            Variable{type}(Wc,true,true,true), Variable{type}(Uc,true,true,true), Variable{type}(bc,true,true,true), nothing)
    end
    function IndGRU()
        new(nothing, nothing, nothing,
            nothing, nothing, nothing,
            nothing, nothing, nothing, nothing)
    end
end


function clone(this::IndGRU; type::Type=Array{Float32})
    cloned = IndGRU()
    cloned.Wz = clone(this.Wz, type=type)
    cloned.bz = clone(this.bz, type=type)
    cloned.Uz = clone(this.Uz, type=type)

    cloned.Wr = clone(this.Wr, type=type)
    cloned.br = clone(this.br, type=type)
    cloned.Ur = clone(this.Ur, type=type)

    cloned.Wc = clone(this.Wc, type=type)
    cloned.bc = clone(this.bc, type=type)
    cloned.Uc = clone(this.Uc, type=type)
    return cloned
end


mutable struct IndGRUs <: Block
    layers::Vector{IndGRU}
    function IndGRUs(topology::Vector{Int}; type::Type=Array{Float32})
        n = length(topology) - 1
        layers = Vector{IndGRU}(undef, n)
        for i = 1:n
            layers[i] = IndGRU(topology[i], topology[i+1]; type=type)
        end
        new(layers)
    end
end


Base.getindex(m::IndGRUs,     k...) =  m.layers[k...]
Base.setindex!(m::IndGRUs, v, k...) = (m.layers[k...] = v)
Base.length(m::IndGRUs)       = length(m.layers)
Base.lastindex(m::IndGRUs)    = length(m.layers)
Base.firstindex(m::IndGRUs)   = 1
Base.iterate(m::IndGRUs, i=firstindex(m)) = i>length(m) ? nothing : (m[i], i+1)


function Base.show(io::IO, m::IndGRU)
    SIZE = size(m.Wr)
    TYPE = typeof(m.Wr.value)
    print(io, "IndGRU($(SIZE[2]), $(SIZE[1]); type=$TYPE)")
end


function Base.show(io::IO, model::IndGRUs)
    for m in model
        show(io, m)
    end
end


function resethidden(model::IndGRU)
    model.h = nothing
end


function resethidden(model::IndGRUs)
    for m in model
        resethidden(m)
    end
end


function forward(model::IndGRU, x::Variable{T}) where T
    Wz = model.Wz
    Uz = model.Uz
    bz = model.bz

    Wr = model.Wr
    Ur = model.Ur
    br = model.br

    Wc = model.Wc
    Uc = model.Uc
    bc = model.bc

    h = model.h ≠ nothing ? model.h : Variable(Zeros(T, size(Wr,1), size(x,2)), type=T)

    z = sigmoid(Wz * x + Uz .* h .+ bz)
    r = sigmoid(Wr * x + Ur .* h .+ br)
    c = tanh(   Wc * x + Uc .* (r .* h) .+ bc )
    h = z .* h + (1 - z) .* c

    model.h = h

    return h
end


function forward(model::IndGRUs, x::Variable)
    for m in model
        x = forward(m, x)
    end
    return x
end


function predict(model::IndGRU, x::T) where T
    Wz = model.Wz.value
    Uz = model.Uz.value
    bz = model.bz.value

    Wr = model.Wr.value
    Ur = model.Ur.value
    br = model.br.value

    Wc = model.Wc.value
    Uc = model.Uc.value
    bc = model.bc.value

    h = model.h ≠ nothing ? model.h : Zeros(T, size(Wr,1), size(x,2))

    z = sigmoid(Wz * x + Uz .* h .+ bz)
    r = sigmoid(Wr * x + Ur .* h .+ br)
    c = tanh(   Wc * x + Uc .* (r .* h) .+ bc )
    h = z .* h + (eltype(T)(1) .- z) .* c

    model.h = h

    return h
end


function predict(model::IndGRUs, x)
    for m in model
        x = predict(m, x)
    end
    return x
end


"""
    unbiasedof(m::IndGRU)

unbiasedof weights of IndGRU block
"""
function unbiasedof(m::IndGRU)
    weights = Vector(undef, 3)
    weights[1] = m.Wz.value
    weights[2] = m.Wr.value
    weights[3] = m.Wc.value
    return weights
end


function weightsof(m::IndGRU)
    weights = Vector{Variable}(undef,9)
    weights[1] = m.Wz.value
    weights[2] = m.Uz.value
    weights[3] = m.bz.value

    weights[4] = m.Wr.value
    weights[5] = m.Ur.value
    weights[6] = m.br.value

    weights[7] = m.Wc.value
    weights[8] = m.Uc.value
    weights[9] = m.bc.value
    return weights
end


"""
    unbiasedof(model::IndGRUs)

unbiased weights of IndGRUs block
"""
function unbiasedof(model::IndGRUs)
    weights = Vector(undef, 0)
    for m in model
        append!(weights, unbiasedof(m))
    end
    return weights
end


function weightsof(model::IndGRUs)
    weights = Vector(undef,0)
    for m in model
        append!(weights, weightsof(m))
    end
    return weights
end


function gradsof(m::IndGRU)
    grads = Vector{Variable}(undef,9)
    grads[1] = m.Wz.delta
    grads[2] = m.Uz.delta
    grads[3] = m.bz.delta

    grads[4] = m.Wr.delta
    grads[5] = m.Ur.delta
    grads[6] = m.br.delta

    grads[7] = m.Wc.delta
    grads[8] = m.Uc.delta
    grads[9] = m.bc.delta
    return grads
end


function gradsof(model::IndGRUs)
    grads = Vector(undef,0)
    for m in model
        append!(grads, gradsof(m))
    end
    return grads
end


function zerograds!(m::IndGRU)
    for v in gradsof(m)
        v .= zero(v)
    end
end


function zerograds!(m::IndGRUs)
    for v in gradsof(m)
        v .= zero(v)
    end
end


function paramsof(m::IndGRU)
    params = Vector{Variable}(undef,9)
    params[1] = m.Wz
    params[2] = m.Uz
    params[3] = m.bz

    params[4] = m.Wr
    params[5] = m.Ur
    params[6] = m.br

    params[7] = m.Wc
    params[8] = m.Uc
    params[9] = m.bc
    return params
end


function xparamsof(m::IndGRU)
    xparams = Vector{XVariable}(undef,9)
    xparams[1] = ('w', m.Wz)
    xparams[2] = ('u', m.Uz)
    xparams[3] = ('b', m.bz)

    xparams[4] = ('w', m.Wr)
    xparams[5] = ('u', m.Ur)
    xparams[6] = ('b', m.br)

    xparams[7] = ('w', m.Wc)
    xparams[8] = ('u', m.Uc)
    xparams[9] = ('b', m.bc)
    return xparams
end


function paramsof(model::IndGRUs)
    params = Vector{Variable}(undef,0)
    for m in model
        append!(params, paramsof(m))
    end
    return params
end


function xparamsof(model::IndGRUs)
    xparams = Vector{XVariable}(undef,0)
    for m in model
        append!(xparams, xparamsof(m))
    end
    return xparams
end


function nparamsof(m::IndGRU)
    lw = length(m.Wr)
    lu = length(m.Ur)
    lb = length(m.br)
    return (lw+lu+lb)*3
end


function bytesof(model::IndGRU, unit::String="MB")
    n = nparamsof(model) * elsizeof(model.Wr)
    return blocksize(n, uppercase(unit))
end


function nparamsof(model::IndGRUs)
    num = 0
    for m in model
        num += nparamsof(m)
    end
    return num
end


function bytesof(model::IndGRUs, unit::String="MB")
    n = nparamsof(model) * elsizeof(model[1].Wr)
    return blocksize(n, uppercase(unit))
end


function to(type::Type, m::IndGRU)
    m.Wz = to(type, m.Wz)
    m.Uz = to(type, m.Uz)
    m.bz = to(type, m.bz)

    m.Wr = to(type, m.Wr)
    m.Ur = to(type, m.Ur)
    m.br = to(type, m.br)

    m.Wc = to(type, m.Wc)
    m.Uc = to(type, m.Uc)
    m.bc = to(type, m.bc)
    return m
end


function to!(type::Type, m::IndGRU)
    m = to(type, m)
    return nothing
end


function to(type::Type, m::IndGRUs)
    for layer in m
        layer = to(type, layer)
    end
    return m
end


function to!(type::Type, m::IndGRUs)
    for layer in m
        to!(type, layer)
    end
end


elsizeof(i::IndGRU) = elsizeof(i.Wr)
elsizeof(i::IndGRUs) = elsizeof(i[1].Wr)
