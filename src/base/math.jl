export emul
export ediv
export addmv
export mulmv
export assert_same_size

@inline function assert_same_size(x::Union{Variable,AbstractArray}, y::Union{Variable,AbstractArray})
    @assert size(x) == size(y) "2 inputs shall be the same size"
end



function Base.:+(x::Variable{T}, constant::Real) where T
    # a matrix add a constant element by element
    C = eltype(ᵛ(x))(constant)
    y = Variable{T}(ᵛ(x) .+ C, x.backprop)
    if y.backprop
        y.backward = function ∇addconst()
            if needgrad(x)
                x ← δ(y)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function Base.:+(constant::Real, var::Variable{T}) where T
    return var + constant
end


function Base.:-(x::Variable{T}, constant::Real) where T
    # a matrix minus a constant element by element
    C = eltype(ᵛ(x))(constant)
    y = Variable{T}(ᵛ(x) .- C, x.backprop)
    if y.backprop
        y.backward = function ∇minusconst()
            if needgrad(x)
                x ← δ(y)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function Base.:-(constant::Real, x::Variable{T}) where T
    # a matrix minus a constant element by element
    C = eltype(ᵛ(x))(constant)
    y = Variable{T}(C .- ᵛ(x), x.backprop)
    if y.backprop
        y.backward = function ∇constminus()
            if needgrad(x)
                x ← - δ(y)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function Base.:-(x::Variable{T}) where T
    # a matrix minus a constant element by element
    y = Variable{T}(- ᵛ(x), x.backprop)
    if y.backprop
        y.backward = function ∇negate()
            if needgrad(x)
                x ← - δ(y)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function Base.:*(x::Variable{T}, constant::Real) where T
    # a matrix multiplies a constant element by element
    C = eltype(ᵛ(x))(constant)
    y = Variable{T}(ᵛ(x) .* C, x.backprop)
    if y.backprop
        y.backward = function ∇mulconst()
            if needgrad(x)
                x ← δ(y) .* C
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function Base.:*(constant::Real, x::Variable{T}) where T
    return x * constant
end

function Base.:/(x::Variable{T}, constant::Real) where T
    return x * (1 / constant)
end


function Base.:^(x::Variable{T}, n::Real) where T
    # 矩阵、列向量与常数按元素做幂指数运算
    n = eltype(ᵛ(x))(n)
    y = Variable{T}(ᵛ(x) .^ n, x.backprop)
    if y.backprop
        y.backward = function ∇power()
            if needgrad(x)
                x ← n .* ᵛ(y) ./ ᵛ(x) .* δ(y)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function Base.:+(x::Variable{T1}, y::Variable{T2}) where {T1,T2}
    assert_same_size(x, y)
    backprop = (x.backprop || y.backprop)
    T = vartype(T1, T2)
    z = Variable{T}(ᵛ(x) + ᵛ(y), backprop)
    if backprop
        z.backward = function ∇eadd()
            needgrad(x) && (x ← δ(z))
            needgrad(y) && (y ← δ(z))
            ifNotKeepδThenFreeδ!(z)
        end
        addchild(z, x)
        addchild(z, y)
    end
    return z
end


function Base.:+(x::Variable{T}, y::AbstractArray) where T
    assert_same_size(x, y)
    z = Variable{T}(ᵛ(x) + y, x.backprop)
    if z.backprop
        z.backward = function ∇eadd()
            if needgrad(x)
                x ← δ(z)
            end
            ifNotKeepδThenFreeδ!(z)
        end
        addchild(z, x)
    end
    return z
end

function Base.:+(x::AbstractArray, y::Variable{T}) where T
    assert_same_size(x, y)
    z = Variable{T}(x + ᵛ(y), y.backprop)
    if z.backprop
        z.backward = function ∇eadd()
            if needgrad(y)
                y ← δ(z)
            end
            ifNotKeepδThenFreeδ!(z)
        end
        addchild(z, y)
    end
    return z
end


function Base.:-(x::Variable{T1}, y::Variable{T2}) where {T1,T2}
    assert_same_size(x, y)
    backprop = (x.backprop || y.backprop)
    T = vartype(T1, T2)
    z = Variable{T}(ᵛ(x) - ᵛ(y), backprop)
    if backprop
        z.backward = function ∇eminus()
            needgrad(x) && (x ←  δ(z))
            needgrad(y) && (y ← -δ(z))
            ifNotKeepδThenFreeδ!(z)
        end
        addchild(z, x)
        addchild(z, y)
    end
    return z
end


function Base.:-(x::Variable{T}, y::AbstractArray) where T
    assert_same_size(x, y)
    z = Variable{T}(ᵛ(x) - y, x.backprop)
    if z.backprop
        z.backward = function ∇eminus()
            if needgrad(x)
                x ← δ(z)
            end
            ifNotKeepδThenFreeδ!(z)
        end
        addchild(z, x)
    end
    return z
end

function Base.:-(x::AbstractArray, y::Variable{T}) where T
    assert_same_size(x, y)
    z = Variable{T}(x - ᵛ(y), y.backprop)
    if z.backprop
        z.backward = function ∇eminus()
            if needgrad(y)
                y ← - δ(z)
            end
            ifNotKeepδThenFreeδ!(z)
        end
        addchild(z, y)
    end
    return z
end


"""
    emul(x::Variable{T1}, y::Variable{T2}) where {T1,T2}
a tensor multiplies a tensor element by element, size(x)==size(y)
"""
function emul(x::Variable{T1}, y::Variable{T2}) where {T1,T2}
    assert_same_size(x, y)
    backprop = (x.backprop || y.backprop)
    T = vartype(T1, T2)
    z = Variable{T}(ᵛ(x) .* ᵛ(y), backprop)
    if backprop
        z.backward = function ∇emul()
            needgrad(x) && (x ← δ(z) .* ᵛ(y))
            needgrad(y) && (y ← δ(z) .* ᵛ(x))
            ifNotKeepδThenFreeδ!(z)
        end
        addchild(z, x)
        addchild(z, y)
    end
    return z
end


"""
    emul(x::Variable, y::AbstractArray)
a tensor multiplies a tensor element by element
"""
function emul(x::Variable{T}, y::AbstractArray) where T
    assert_same_size(x, y)
    z = Variable{T}(ᵛ(x) .* y, x.backprop)
    if backprop
        z.backward = function ∇emul()
            if needgrad(x)
                x ← δ(z) .* ᵛ(y)
            end
            ifNotKeepδThenFreeδ!(z)
        end
        addchild(z, x)
    end
    return z
end


"""
    emul(x::AbstractArray, y::Variable)
a tensor multiplies a tensor element by element
"""
function emul(x::AbstractArray, y::Variable{T}) where T
    assert_same_size(x, y)
    z = Variable{T}(x .* ᵛ(y), x.backprop)
    if backprop
        z.backward = function ∇emul()
            if needgrad(y)
                y ← δ(z) .* ᵛ(x)
            end
            ifNotKeepδThenFreeδ!(z)
        end
        addchild(z, y)
    end
    return z
end


"""
    ediv(x::Variable{T1}, y::Variable{T2}) where {T1,T2}
a tensor divided by a tensor element by element, size(`x`)==size(`y`)
"""
function ediv(x::Variable{T1}, y::Variable{T2}) where {T1,T2}
    assert_same_size(x, y)
    T = vartype(T1, T2)
    z = Variable{T}(ᵛ(x) ./ ᵛ(y), x.backprop || y.backprop)
    if z.backprop
        z.backward = function ∇ediv()
            δx = δ(z) ./ ᵛ(y)
            if needgrad(x)
                x ← δx
            end
            if needgrad(y)
                y ← - δx .* ᵛ(z)
            end
            ifNotKeepδThenFreeδ!(z)
        end
        addchild(z, x)
        addchild(z, y)
    end
    return z
end


function ediv(x::Variable{T}, y::AbstractArray) where T
    assert_same_size(x, y)
    z = Variable{T}(ᵛ(x) ./ y, x.backprop)
    if z.backprop
        z.backward = function ∇ediv()
            if needgrad(x)
                x ← δ(z) ./ y
            end
            ifNotKeepδThenFreeδ!(z)
        end
        addchild(z, x)
    end
    return z
end

function ediv(x::AbstractArray, y::Variable{T}) where T
    assert_same_size(x, y)
    z = Variable{T}(x ./ ᵛ(y), x.backprop || y.backprop)
    if z.backprop
        z.backward = function ∇ediv()
            if needgrad(y)
                y ← - δ(z) ./ ᵛ(y) .* ᵛ(z)
            end
            ifNotKeepδThenFreeδ!(z)
        end
        addchild(z, y)
    end
    return z
end

function Base.:*(W::Variable{T1}, X::Variable{T2}) where {T1,T2}
    # matrix W multiplies matrix X
    # 矩阵相乘 Y[i,j] = sum(W[i,k]*X[k,j],k=...)
    # W -- 权重矩阵
    # X -- n个输入列向量组成的矩阵
    # Y -- n个输出列向量组成的矩阵
    backprop = (W.backprop || X.backprop)
    T = vartype(T1, T2)
    Y = Variable{T}(ᵛ(W) * ᵛ(X), backprop)
    if backprop
        Y.backward = function ∇matMul()
            needgrad(W) && (W ← δ(Y)  * ᵛ(X)')
            needgrad(X) && (X ← ᵛ(W)' * δ(Y) )
            ifNotKeepδThenFreeδ!(Y)
        end
        addchild(Y, W)
        addchild(Y, X)
    end
    return Y
end


function Base.:*(W::Variable{T}, X::AbstractArray) where T
    Y = Variable{T}(ᵛ(W) * X, W.backprop)
    if Y.backprop
        Y.backward = function ∇matMul()
            if needgrad(W)
                W ← δ(Y) * X'
            end
            ifNotKeepδThenFreeδ!(Y)
        end
        addchild(Y, W)
    end
    return Y
end


function Base.:*(W::AbstractArray, X::Variable{T}) where T
    Y = Variable{T}(W * ᵛ(X), X.backprop)
    if Y.backprop
        Y.backward = function ∇matMul()
            if needgrad(X)
                X ← W' * δ(Y)
            end
            ifNotKeepδThenFreeδ!(Y)
        end
        addchild(Y, X)
    end
    return Y
end


"""
    addmv(m::Variable{T1}, v::Variable{T2}) where {T1,T2}
a matrix tensor `m` adds a vector tensor `v`
"""
function addmv(M::Variable{T1}, V::Variable{T2}) where {T1,T2}
    @assert (M.shape[1]==V.shape[1] && V.shape[2]==1)
    backprop = (M.backprop || V.backprop)
    T = vartype(T1, T2)
    Z = Variable{T}(ᵛ(M) .+ ᵛ(V), backprop)
    if backprop
        Z.backward = function ∇addmv()
            needgrad(M) && (M ← δ(Z))
            needgrad(V) && (V ← sum(δ(Z), dims=2))
            ifNotKeepδThenFreeδ!(Z)
        end
        addchild(Z, M)
        addchild(Z, V)
    end
    return Z
end


"""
    mulmv(m::Variable{T1}, v::Variable{T2}) where {T1,T2}
a matrix tensor `m` multiplies a vector tensor `v`
"""
function mulmv(M::Variable{T1}, V::Variable{T2}) where {T1,T2}
    @assert (M.shape[1]==V.shape[1] && V.shape[2]==1)
    backprop = (M.backprop || V.backprop)
    T = vartype(T1, T2)
    Z = Variable{T}(ᵛ(M) .* ᵛ(V), backprop)
    if backprop
        Z.backward = function ∇mulmv()
            needgrad(M) && (M ←     δ(Z) .* ᵛ(V))
            needgrad(V) && (V ← sum(δ(Z) .* ᵛ(M), dims=2))
            ifNotKeepδThenFreeδ!(Z)
        end
        addchild(Z, M)
        addchild(Z, V)
    end
    return Z
end


export GradScalar
mutable struct GradScalar
    v::Real
    function GradScalar(v::Real)
        new(v)
    end
end

function Base.:*(x::Variable{T}, constant::GradScalar) where T
    C = eltype(ᵛ(x))(constant.v)
    y = Variable{T}(ᵛ(x), x.backprop)
    if y.backprop
        y.backward = function ∇mulconst()
            if needgrad(x)
                x ← δ(y) .* C
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function Base.:*(constant::GradScalar, var::Variable{T}) where T
    return var * constant
end
