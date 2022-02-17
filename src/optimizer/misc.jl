export clip
export update!
export zerograds!


function clip(x, clipval)
    x = (abs(x) > clipval) ? clipval * sign(x) : x
end


function update!(var::Variable, lr)
    # update single Variable
    @. var.value -= lr * setNanInfZero(var.delta)
end


function update!(vars::Vector{Variable}, lr)
    # update multi Variables
    for var in vars
        update!(var, lr)
    end
end


function update!(vars::Vector{XVariable}, lr)
    # update multi Variables
    for xvar in xparams
        c , x = xvar
        update!(x, lr)
    end
end


function zerograds!(params::Vector{Variable})
    for x in params
        if !isnothing(δ(x))
            δ(x) .= 0.0
        end
    end
end


function zerograds!(xparams::Vector{XVariable})
    for xvar in xparams
        c , x = xvar
        if !isnothing(δ(x))
            δ(x) .= 0.0
        end
    end
end


function zerograds!(O::Optimizer)
    for xvar in O.xparams
        c , x = xvar
        if !isnothing(δ(x))
            δ(x) .= 0.0
        end
    end
end
