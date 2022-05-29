export dotAdd
export dotMul
export matAddVec
export matMulVec


function Base.:+(x::Variable{T}, constant) where T
    # a matrix add a constant element by element
    C = eltype(ᵛ(x))(constant)
    y = Variable{T}(ᵛ(x) .+ C, x.backprop)
    if y.backprop
        y.backward = function ∇matAddScalar()
            if need2computeδ!(x)
                δ(x) .+= δ(y)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function Base.:+(constant, var::Variable{T}) where T
    return var + constant;
end


function Base.:-(x::Variable{T}, constant) where T
    # a matrix minus a constant element by element
    C = eltype(ᵛ(x))(constant)
    y = Variable{T}(ᵛ(x) .- C, x.backprop)
    if y.backprop
        y.backward = function ∇matMinusScalar()
            if need2computeδ!(x)
                δ(x) .+= δ(y)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function Base.:-(constant, x::Variable{T}) where T
    # a matrix minus a constant element by element
    C = eltype(ᵛ(x))(constant)
    y = Variable{T}(C .- ᵛ(x), x.backprop)
    if y.backprop
        y.backward = function ∇scalarMinusMat()
            if need2computeδ!(x)
                δ(x) .-= δ(y)
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
        y.backward = function ∇setNegative()
            if need2computeδ!(x)
                δ(x) .-= δ(y)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function Base.:*(x::Variable{T}, constant) where T
    # a matrix multiplies a constant element by element
    C = eltype(ᵛ(x))(constant)
    y = Variable{T}(ᵛ(x) .* C, x.backprop)
    if y.backprop
        y.backward = function ∇matMulScalar()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* C
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function Base.:*(constant, var::Variable{T}) where T
    return var * constant
end


function Base.:^(x::Variable{T}, n::Real) where T
    # 矩阵、列向量与常数按元素做幂指数运算
    n = eltype(ᵛ(x))(n)
    y = Variable{T}(ᵛ(x) .^ n, x.backprop)
    if y.backprop
        y.backward = function ∇power()
            if need2computeδ!(x)
                δ(x) .+= n .* ᵛ(y) ./ ᵛ(x) .* δ(y)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function Base.:+(x::Variable{T1}, y::Variable{T2}) where {T1,T2}
    # a matrix add a matrix element by element: z = x + y
   T = vartype(T1, T2)
   @assert (x.shape == y.shape) "2 inputs shall be the same size"
   backprop = (x.backprop || y.backprop)
   z = Variable{T}(ᵛ(x) + ᵛ(y), backprop)
   if backprop
       z.backward = function ∇add2var()
           if need2computeδ!(x) δ(x) .+= δ(z) end
           if need2computeδ!(y) δ(y) .+= δ(z) end
           ifNotKeepδThenFreeδ!(z)
       end
       addchild(z, x)
       addchild(z, y)
   end
   return z
end


function Base.:-(x::Variable{T1}, y::Variable{T2}) where {T1,T2}
    # a matrix minus a matrix element by element : z = x - y
    @assert (x.shape == y.shape) "2 inputs shall be the same size"
    backprop = (x.backprop || y.backprop)
    T = vartype(T1, T2)
    z = Variable{T}(ᵛ(x) - ᵛ(y), backprop)
    if backprop
        z.backward = function ∇minus2var()
            if need2computeδ!(x) δ(x) .+= δ(z) end
            if need2computeδ!(y) δ(y) .-= δ(z) end
            ifNotKeepδThenFreeδ!(z)
        end
        addchild(z, x)
        addchild(z, y)
    end
    return z
end


"""
    dotAdd(var1::Variable{T1}, var2::Variable{T2}) where {T1,T2}
a tensor add a tensor element by element
"""
function dotAdd(x::Variable{T1}, y::Variable{T2}) where {T1,T2}
    # a tensor add a tensor element by element: z = x .+ y
    @assert (x.shape == y.shape) "2 inputs shall be the same size"
    backprop = (x.backprop || y.backprop)
    T = vartype(T1, T2)
    z = Variable{T}(ᵛ(x) .+ ᵛ(y), backprop)
    if backprop
        z.backward = function ∇dotAdd()
            if need2computeδ!(x) δ(x) .+= δ(z) end
            if need2computeδ!(y) δ(y) .+= δ(z) end
            ifNotKeepδThenFreeδ!(z)
        end
        addchild(z, x)
        addchild(z, y)
    end
    return z
end


"""
    dotMul(var1::Variable{T1}, var2::Variable{T2}) where {T1,T2}
a tensor multiplies a tensor element by element
"""
function dotMul(x::Variable{T1}, y::Variable{T2}) where {T1,T2}
    # a tensor multiplies a tensor element by element: z = x .* y
    @assert (x.shape == y.shape) "2 inputs shall be the same size"
    backprop = (x.backprop || y.backprop)
    T = vartype(T1, T2)
    z = Variable{T}(ᵛ(x) .* ᵛ(y), backprop)
    if backprop
        z.backward = function ∇dotMul()
            if need2computeδ!(x) δ(x) .+= δ(z) .* ᵛ(y) end
            if need2computeδ!(y) δ(y) .+= δ(z) .* ᵛ(x) end
            ifNotKeepδThenFreeδ!(z)
        end
        addchild(z, x)
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
            if need2computeδ!(W) δ(W) .+= δ(Y)  * ᵛ(X)' end
            if need2computeδ!(X) δ(X) .+= ᵛ(W)' * δ(Y)  end
            ifNotKeepδThenFreeδ!(Y)
        end
        addchild(Y, W)
        addchild(Y, X)
    end
    return Y
end


"""
    matAddVec(var1::Variable{T1}, var2::Variable{T2}) where {T1,T2}
a matrix tensor `var1` adds a vector tensor `var2`
"""
function matAddVec(M::Variable{T1}, V::Variable{T2}) where {T1,T2}
    # M -- 充当和节点，非学习的参数
    # V -- 偏置列向量，要学习的参数
    # Z = M .+ V
    @assert (M.shape[1]==V.shape[1] && V.shape[2]==1)
    backprop = (M.backprop || V.backprop)
    T = vartype(T1, T2)
    Z = Variable{T}(ᵛ(M) .+ ᵛ(V), backprop)
    if backprop
        Z.backward = function ∇matAddVec()
            if need2computeδ!(M) δ(M) .+=     δ(Z)          end
            if need2computeδ!(V) δ(V) .+= sum(δ(Z), dims=2) end
            ifNotKeepδThenFreeδ!(Z)
        end
        addchild(Z, M)
        addchild(Z, V)
    end
    return Z
end


"""
    matAddVec(var1::Variable{T1}, var2::Variable{T2}) where {T1,T2}
a matrix tensor `var1` multiplies a vector tensor `var2`
"""
function matMulVec(M::Variable{T1}, V::Variable{T2}) where {T1,T2}
    # M -- 一般充当激活节点，非网络需要学习的参数
    # V -- 列向量，循环权重，是网络需要学习的参数
    # Z = M .* V
    @assert (M.shape[1]==V.shape[1] && V.shape[2]==1)
    backprop = (M.backprop || V.backprop)
    T = vartype(T1, T2)
    Z = Variable{T}(ᵛ(M) .* ᵛ(V), backprop)
    if backprop
        Z.backward = function ∇matMulVec()
            if need2computeδ!(M) δ(M) .+=     δ(Z) .* ᵛ(V)          end
            if need2computeδ!(V) δ(V) .+= sum(δ(Z) .* ᵛ(M), dims=2) end
            ifNotKeepδThenFreeδ!(Z)
        end
        addchild(Z, M)
        addchild(Z, V)
    end
    return Z
end
