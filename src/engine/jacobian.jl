export jacobian


"""
    jacobian(y::Variable{T}, x::Variable{T}) where T <: AbstractArray

J = ∂y/∂x , if y ∈ Rᵐ and x ∈ Rⁿ , then J ∈ Rᵐⁿ, so J is a matrix \n
grouped by m-row gradient together.
"""
function jacobian(y::Variable{T}, x::Variable{T}) where T <: AbstractArray
    # suppose y ∈ Rᵐ and x ∈ Rⁿ , then J ∈ Rᵐⁿ
    S = eltype(y)
    m = length(y)
    n = length(x)
    J = T(undef, m, n)

    if need2computeδ!(y)
        if y.indegree == 0
            sorted = sort_by_dfs(y)
        else
            resetindegree(y)
            sorted = sort_by_dfs(y)
        end
    end

    x.keepsgrad = true
    y.keepsgrad = true
    zerodelta(x)
    o = S(0)
    l = S(1)
    
    for i = 1:m
        x.delta .= o    # x.delta is accumulated each time so shall be set zero
        y.delta .= o    # set all elements of y as zero expect
        y.delta[i] = l  # the i-th element as one
        backprop(sorted, x)
        J[i,:] = reshape(x.delta, 1, n)
    end
    return J
end
