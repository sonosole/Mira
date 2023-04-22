export clip
export update!
export zerograds!


function decay_by_L₁(c::Char)
    return c == 'w' ? true : false
end

function decay_by_L₂(c::Char)
    return c == 'w' ? true : false
end


function clip(x::Real, clipval::Real)
    x = (abs(x) > clipval) ? clipval * sign(x) : x
end

function clip(x::AbstractArray, clipval::Real)
    i = abs.(x) .> clipval
    x[i] .= clipval .* sign(x[i])
    return x
end

function update!(x::Variable, lr::AbstractFloat)
    # update single Variable
    if !isnothing(δ(x))
        setNanInfZero!(δ(x))
        ᵛ(x) .-= lr .* δ(x)
    end
end


function update!(vars::Vector{Variable}, lr::AbstractFloat)
    # update multi Variables
    Threads.@threads for var in vars
        update!(var, lr)
    end
end


function update!(xparams::Vector{XVariable}, lr::AbstractFloat)
    # update multi Variables
    Threads.@threads for xvar in xparams
        c , x = xvar
        update!(x, lr)
    end
end


function zerograds!(x::Variable)
    if !isnothing(δ(x))
        δ(x) .= 0.0
    end
end


function zerograds!(params::Vector{Variable})
    Threads.@threads for x in params
        if !isnothing(δ(x))
            δ(x) .= 0.0
        end
    end
end


function zerograds!(xp::XVariable)
    c , x = xp
    if !isnothing(δ(x))
        δ(x) .= 0.0
    end
end


function zerograds!(xparams::Vector{XVariable})
    Threads.@threads for xvar in xparams
        c , x = xvar
        if !isnothing(δ(x))
            δ(x) .= 0.0
        end
    end
end


function zerograds!(O::Optimizer)
    Threads.@threads for xvar in O.xparams
        c , x = xvar
        if !isnothing(δ(x))
            δ(x) .= 0.0
        end
    end
end
