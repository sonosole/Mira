"""
# mutable struct ThresholdSinceLast
    If the loss decrement was lower than a given threshold compared to the last loss, then trigger.
# Constructor
    ThresholdSinceLast(threshold::Real, lastloss::Real=Inf)
# Example
    julia> p = ThresholdSinceLast(1.0);
    julia> for loss in [7, 7, 8, 6, 7, 9]
            if actnow(p, loss)
                println(loss);break
            end
        end
    8
"""
mutable struct ThresholdSinceLast
    threshold :: Real  # if decrement < threshold then trigger
    lastloss  :: Real  # last best loss recorded
    function ThresholdSinceLast(threshold::Real, lastloss::Real=Inf)
        @assert threshold > 0 "threshold is positive, but got $threshold"
        return new(threshold, lastloss)
    end
end


function Base.show(io::IO, t::ThresholdSinceLast)
    ϴ = t.threshold
    l = t.lastloss
    print("ThresholdSinceLast(threshold=$ϴ, lastloss=$l)")
end


function trigger(t::ThresholdSinceLast, loss::Real)
    decrement  = t.lastloss - loss
    t.lastloss = loss
    if decrement ≥ t.threshold
        return false
    else
        return true
    end
end
