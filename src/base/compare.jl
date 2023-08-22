export elements_great_than_zero
export elements_great_equal_than_zero

export elements_less_than_zero
export elements_less_equal_than_zero


"""
    elements_great_than_zero(x::Variable) -> y::Variable

Elements great than 0 are kept, others set 0
"""
function elements_great_than_zero(x::Variable{T}) where T
    o = eltype(T)(0)  # typed zero
    m = ᵛ(x) .> o     # mask that meets requirement
    y = Variable{T}(ᵛ(x) .* m, x.backprop)
    if y.backprop
        y.backward = function ∇elements_great_than_zero()
            if need2computeδ!(x)
                x ← δ(y) .* m
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


"""
    elements_great_equal_than_zero(x::Variable) -> y::Variable

Elements great equal than 0 are kept, others set 0
"""
function elements_great_equal_than_zero(x::Variable{T}) where T
    o = eltype(T)(0)  # typed zero
    m = ᵛ(x) .≥ o     # mask that meets requirement
    y = Variable{T}(ᵛ(x) .* m, x.backprop)
    if y.backprop
        y.backward = function ∇elements_great_equal_than_zero()
            if need2computeδ!(x)
                x ← δ(y) .* m
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end




"""
    elements_less_than_zero(x::Variable) -> y::Variable

Elements less than 0 are kept, others set 0
"""
function elements_less_than_zero(x::Variable{T}) where T
    o = eltype(T)(0)  # typed zero
    m = ᵛ(x) .< o     # mask that meets requirement
    y = Variable{T}(ᵛ(x) .* m, x.backprop)
    if y.backprop
        y.backward = function ∇elements_less_than_zero()
            if need2computeδ!(x)
                x ← δ(y) .* m
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


"""
    elements_less_equal_than_zero(x::Variable) -> y::Variable

Elements less equal than 0 are kept, others set 0
"""
function elements_less_equal_than_zero(x::Variable{T}) where T
    o = eltype(T)(0)  # typed zero
    m = ᵛ(x) .≤ o     # mask that meets requirement
    y = Variable{T}(ᵛ(x) .* m, x.backprop)
    if y.backprop
        y.backward = function ∇elements_less_equal_than_zero()
            if need2computeδ!(x)
                x ← δ(y) .* m
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


# x > y is a constraint, if satisfied then no cost paid, else it has cost (correspond to x ≤ y)
(>)(x::Variable, y::Variable) = elements_less_equal_than_zero(x - y)

# x ≥ y is a constraint, if satisfied then no cost paid, else it has cost (correspond to x < y)
(≥)(x::Variable, y::Variable) = elements_less_than_zero(x - y)

# x < y is a constraint, if satisfied then no cost paid, else it has cost (correspond to x ≥ y)
(<)(x::Variable, y::Variable) = elements_great_equal_than_zero(x - y)

# x ≤ y is a constraint, if satisfied then no cost paid, else it has cost (correspond to x > y)
(≤)(x::Variable, y::Variable) = elements_great_than_zero(x - y)
