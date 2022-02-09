mutable struct SelfLoopCumulativeResNet <: Block
    self::Union{Block,Nothing}
    degree::Int
    function SelfLoopCumulativeResNet(self::Block, n::Int)
        new(self, n)
    end
    function SelfLoopCumulativeResNet(n::Int)
        new(nothing, n)
    end
end


function forward(m::SelfLoopCumulativeResNet, x::Variable)
    for i = 1:m.degree
        x = forward(m.self, x) + x
    end
    return x
end


function predict(m::SelfLoopCumulativeResNet,  x::AbstractArray)
    for i = 1:m.degree
        x = predict(m.self, x) + x
    end
    return x
end


function clone(this::SelfLoopCumulativeResNet; type::Type=Array{Float32})
    cloned = SelfLoopCumulativeResNet(this.degree)
    cloned.self = clone(this.self, type=type)
    return cloned
end


unbiasedof(m::SelfLoopCumulativeResNet) = unbiasedof(m.self)
weightsof(m::SelfLoopCumulativeResNet)  = weightsof(m.self)
gradsof(m::SelfLoopCumulativeResNet)    = gradsof(m.self)
zerograds!(m::SelfLoopCumulativeResNet) = zerograds!(m.self)
paramsof(m::SelfLoopCumulativeResNet)  = paramsof(m.self)
xparamsof(m::SelfLoopCumulativeResNet) = xparamsof(m.self)
nparamsof(m::SelfLoopCumulativeResNet) = nparamsof(m.self)
bytesof(m::SelfLoopCumulativeResNet)   = bytesof(m.self)


function to(type::Type, m::SelfLoopCumulativeResNet)
    m.self = to(type, m.self)
    return m
end


function to!(type::Type, m::SelfLoopCumulativeResNet)
    m.self = to(type, m.self)
    return nothing
end
