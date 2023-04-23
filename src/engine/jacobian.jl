export jacobian


"""
    jacobian(y::Variable{T}, x::Variable{T}) where T <: AbstractArray

J = ∂y/∂x , if y ∈ Rᵐ and x ∈ Rⁿ , then J ∈ Rᵐⁿ, so J is a matrix \n
grouped by m-row gradient together.
"""
function jacobian(y::Variable{T}, x::Variable{T}) where T <: AbstractArray
    S = eltype(y)
    o = S(0)
    l = S(1)

    m = length(y)
    n = length(x)
    J = Zeros(T, m, n)

    # note: x is not included in sorted
    sorted = sort_by_dfs(y, x)
    for node in sorted
        node.keepsgrad = true
        node.delta = zero(node.value)
    end
    x.keepsgrad = true
    x.delta = zero(x.value)

    for i = 1:m
        # δy is one hot vector
        y.delta[i] = l
        backprop(sorted)
        J[i,:] = reshape(x.delta, 1, n)

        # node.delta is accumulated each time so shall be set zero
        for node in sorted
            node.delta .= o
        end
        # x.delta is accumulated each time so shall be set zero
        x.delta .= o
    end
    return J
end
