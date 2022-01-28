export Variable
export zeroDelta
export Zeros, Ones
export clone
export need2computeδ!
export ifNotKeepδThenFreeδ!
export elsizeof
export value, delta, ᵛ, ᵟ, δ

export XVariable, VarOrNil, FunOrNil
const FunOrNil = Union{Function, Nothing}





"""
    mutable struct Variable{T}
# Fields
+ `value::T`                : value in forward
+ `delta::Union{Nothing,T}` : gradients collected in backprop
+ `shape::Tuple`            : shape of `value`
+ `keepsgrad::Bool`         : whether keeps grad after backprop
+ `isleaf::Bool`            : whether leaf node
+ `backward::FunOrNil`      : backward function
"""
mutable struct Variable{T}
    value::T
    delta::Union{Nothing,T}
    shape::Tuple
    backprop::Bool      # 是否反向传播，依赖的子节点都跟其一样
    keepsgrad::Bool     # 是否保留梯度，可能用于训练或者其他
    isleaf::Bool        # 可训练的，代表可学习的参数
    backward::FunOrNil  # 反向传播函数
    function Variable{T}(x, backprop::Bool=true,
                            keepsgrad::Bool=false,
                            isleaf::Bool=false) where T
        s = size(x)
        δ = nothing
        new{T}(x, δ, s, backprop, keepsgrad, isleaf, nothing)
    end
end


# Convenient abstract-type-specilized constructors for data on GPU/CPU/xPU etc....
function Variable(x; backprop::Bool=true,
                     keepsgrad::Bool=false,
                     type::Type=Array{Float32})
    isleaf = true    # any user defined Variable is a leaf Variable
    return Variable{type}(x, backprop, keepsgrad, isleaf)
end


const XVariable = Tuple{Char, Variable}
const VarOrNil  = Union{Variable, Nothing}


# pretty printing
function Base.show(io::IO, x::Variable{T}) where T
    if  x.isleaf println(cyan!("\n≡≡≡ Leaf Variable ≡≡≡")) end
    if !x.isleaf println(cyan!("\n≡≡≡ None Leaf Variable ≡≡≡")) end

    print(blue!("\nvalue is "))
    display(x.value)
    print(green!("\ndelta is "))
    display(x.delta)
end


function clone(x::Variable; type::Type=Array{Float32})
    return Variable{type}(type(x.value), x.backprop, x.keepsgrad, x.isleaf)
end


function zeroDelta(x::Variable{T}) where T
    # 要切断某些反向传播路径的时候将其初始化为零
    if isnothing(x.delta)
        x.delta = Zeros(T, x.shape);
    end
end


function need2computeδ!(x::Variable{T}) where T
    # 不需要学习的叶子参数不需要初始化，其他情况都要
    if !(x.isleaf && !x.keepsgrad)
        if isnothing(x.delta)
            x.delta = Zeros(T, x.shape);
        end
        return true
    else
        return false
    end
end


function ifNotKeepδThenFreeδ!(x::Variable{T}) where T
    if !x.keepsgrad
        x.delta = nothing
    end
end



Base.sizeof(x::Variable)         =  sizeof(x.value)
Base.size(x::Variable)           =    size(x.value)
Base.size(x::Variable, dim::Int) =    size(x.value, dim)
Base.ndims(x::Variable)          =   ndims(x.value)
Base.length(x::Variable)         =  length(x.value)
Base.strides(x::Variable)        = strides(x.value)
Base.eltype(x::Variable)         =  eltype(x.value)
Base.similar(x::Variable{T})  where T = Variable{T}( similar(x.value), x.backprop, x.keepsgrad, x.isleaf)
Base.copy(x::Variable{T})     where T = Variable{T}(    copy(x.value), x.backprop, x.keepsgrad, x.isleaf)
Base.deepcopy(x::Variable{T}) where T = Variable{T}(deepcopy(x.value), x.backprop, x.keepsgrad, x.isleaf)

Base.setindex!(x::Variable, v::Number,        k...) = (x.value[k...] .= v)
Base.setindex!(x::Variable, v::AbstractArray, k...) = (x.value[k...]  = v)



@inline function backward!(x::Variable)
    if !x.isleaf && !isnothing(x.backward)
        return x.backward(ᵟ(x))
    end
end

function Base.getindex(x::Variable{T}, k...) where T
    y = Variable{T}(x.value[k...], x.backprop, x.keepsgrad, x.isleaf)
    if y.backprop
        y.backward = function getindexBackward(δy)
            if need2computeδ!(x)
                x.delta[k...] .+= δy
            end
            ifNotKeepδThenFreeδ!(y);
            backward!(x)
        end
    end
    return y
end


function (v::Variable)(i...)
    if v.delta ≠ nothing
        return v.delta[i...]
    else
        return nothing
    end
end



# pretty printing
function Base.show(io::IO, xv::XVariable)
    c, x = xv
    if  x.isleaf println(cyan("\n≡≡≡ Leaf Variable ($c) ≡≡≡")) end
    if !x.isleaf println(cyan("\n≡≡≡ None Leaf Variable ≡≡≡")) end

    print(blue("\nvalue is "))
    display(x.value)
    print(green("\ndelta is "))
    display(x.delta)
end


elsizeof(x::Variable) = sizeof(eltype(x))


# lazy showing way of Variable's main vars
@inline ᵛ(x::Variable) = x.value
@inline ᵟ(x::Variable) = x.delta
@inline δ(x::Variable) = x.delta
@inline value(x::Variable) = x.value
@inline delta(x::Variable) = x.delta
