mutable struct SelfLoopResNet <: Block
    self::Union{Block,Nothing}
    degree::Int
    function SelfLoopResNet(self::Block, n::Int)
        new(self, n)
    end
    function SelfLoopResNet(n::Int)
        new(nothing, n)
    end
end


function forward(m::SelfLoopResNet, x::Variable)
    z = x
    for i = 1:m.degree
        z = forward(m.self, z) + x
    end
    return z
end


function predict(m::SelfLoopResNet,  x::AbstractArray)
    z = x
    for i = 1:m.degree
        z = predict(m.self, z) + x
    end
    return z
end


function clone(this::SelfLoopResNet; type::Type=Array{Float32})
    cloned = SelfLoopResNet(this.degree)
    cloned.self = clone(this.self, type=type)
    return cloned
end


unbiasedof(m::SelfLoopResNet) = unbiasedof(m.self)
weightsof(m::SelfLoopResNet)  = weightsof(m.self)
gradsof(m::SelfLoopResNet)    = gradsof(m.self)
zerograds!(m::SelfLoopResNet) = zerograds!(m.self)
paramsof(m::SelfLoopResNet)  = paramsof(m.self)
xparamsof(m::SelfLoopResNet) = xparamsof(m.self)
nparamsof(m::SelfLoopResNet) = nparamsof(m.self)
bytesof(m::SelfLoopResNet)   = bytesof(m.self)


function to(type::Type, m::SelfLoopResNet)
    m.self = to(type, m.self)
    return m
end


function to!(type::Type, m::SelfLoopResNet)
    m.self = to(type, m.self)
    return nothing
end
