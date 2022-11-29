"""
# mutable struct PatienceSinceLast
    If the loss grows up for a given consecutive counts, then trigger.
# Constructor
    PatienceSinceLast(n::Int)
# Example
    julia> p = PatienceSinceLast(2);
    julia> for loss in [7, 7, 8, 6, 7, 9]
            if actnow(p, loss)
                println(loss);break
            end
        end
    9
"""
mutable struct PatienceSinceLast
    patience :: Int   # if reached then stop
    counter  :: Int   # counter for bad loss
    lastloss :: Real  # last loss recorded
    function PatienceSinceLast(n::Int=1)
        @assert n > 0 "patience is positive, but got $n"
        return new(n, 0, Inf)
    end
end


function Base.show(io::IO, p::PatienceSinceLast)
    ϴ = p.patience
    c = p.counter
    l = p.lastloss
    print("PatienceSinceLast(patience=$ϴ, counter=$c, lastloss=$l)")
end


function trigger(p::PatienceSinceLast, loss::Real)
    if loss > p.lastloss
        p.counter += 1
    else
        p.counter = 0
    end
    p.lastloss = loss
    if p.counter < p.patience
        return false
    else
        p.counter = 0
        return true
    end
end



"""
# mutable struct PatienceSinceBest
    If the loss does NOT get improved up for a given accumulative counts, then trigger.
    Here the "improved" means smaller.
# Constructor
    PatienceSinceBest(n::Int)
# Example
    julia> p = PatienceSinceBest(3);
    julia> for loss in [7, 7, 9, 8, 7]
            if actnow(p, loss)
                println(loss);break
            end
        end
    8
"""
mutable struct PatienceSinceBest
    patience :: Int  # if reached then stop
    counter  :: Int  # counter for bad loss
    bestloss  :: Real # last best loss recorded
    function PatienceSinceBest(n::Int, bestloss::Real=Inf)
        @assert n > 0 "patience is positive, but got $n"
        return new(n, 0, bestloss)
    end
end


function Base.show(io::IO, p::PatienceSinceBest)
    ϴ = p.patience
    c = p.counter
    l = p.bestloss
    print("PatienceSinceBest(patience=$ϴ, counter=$c, lastloss=$l)")
end


function trigger(p::PatienceSinceBest, loss::Real)
    if loss < p.bestloss
        p.bestloss = loss
        p.counter  = 0
    else
        p.counter += 1
    end

    if p.counter < p.patience
        return false
    else
        return true
    end
end
