export checkgrad


function iseq(x::Real, y::Real; tol::AbstractFloat=0.05)
    (isnan(x) || isinf(x)) && return false
    (isnan(y) || isinf(y)) && return false
    sign(x) ≠ sign(y)      && return false
    return abs(x-y) ≤ tol*max(abs(x), abs(y))
end


"""
    checkgrad(block::Block,            # a block contains params
              x::Variable;             # input of block
              eps::AbstractFloat=1e-7, # increment of each params to check gradient
              tol::AbstractFloat=0.1,  # tolerance of error
              onlyone::Bool=false)     # true for only one param shall be checked

if all gradients were true, then it returns true.
"""
function checkgrad(block::B, x::Variable, i::Int=1;
                   eps::AbstractFloat=1e-7,
                   tol::AbstractFloat=0.05,
                   show::Bool=false,
                   onlyone::Bool=false) where B <: Block
    dw = eps
    istrue = true
    params = paramsof(block)
    for w in params
        # [1] forward 1st time
        y₁ = forward(block, x)
        C₁ = Loss(y₁)
        backward(C₁)
        w̄₁ = δ(w)[i]
        zerograds!(params)

        # [2] with a small change
        ᵛ(w)[1] += dw

        # [3] forward 2nd time
        y₂ = forward(block, x)
        C₂ = Loss(y₂)
        backward(C₂)
        w̄₂ = δ(w)[i]
        zerograds!(params)

        # [4] backward gradient vs numerical gradient
        ∂L∂w = (w̄₂ + w̄₁) / 2
        dLdw = (cost(C₂) - cost(C₁)) / dw

        # [5] check if the auto-grad is true or not
        istrue = iseq(∂L∂w, dLdw, tol=tol) && istrue
        if !istrue
            println(yellow!("backward  gradient: $∂L∂w"))
            println(yellow!("numerical gradient: $dLdw"))
        end
        if show
            println("backward  gradient: $∂L∂w")
            println("numerical gradient: $dLdw\n")
        end
        onlyone ? break : continue
    end
    return istrue
end


"""
checkgrad(fn::Function,
          x::Variable;
          eps::AbstractFloat=1e-8,
          tol::AbstractFloat=0.1)

if gradients were true, then it returns true. Attention, functions with ! ending
can NOT used here.
"""
function checkgrad(fn::Function, x::Variable, i::Int=1;
                   show::Bool=false,
                   eps::AbstractFloat=1e-7,
                   tol::AbstractFloat=0.05)
    dx = eps
    x.keepsgrad = true
    # [1] forward 1st time
    y₁ = fn(x)
    C₁ = Loss(y₁)
    backward(C₁)
    x̄₁ = δ(x)[i]
    zerograds!(x)

    # [2] with a small change
    ᵛ(x)[1] += dx

    # [3] forward 2nd time
    y₂ = fn(x)
    C₂ = Loss(y₂)
    backward(C₂)

    x̄₂ = δ(x)[i]
    zerograds!(x)

    # [4] backward gradient vs numerical gradient
    ∂L∂x = (x̄₂ + x̄₁) / 2
    dLdx = (cost(C₂) - cost(C₁)) / dx

    # [5] check if the auto-grad is true or not
    istrue = iseq(∂L∂x, dLdx, tol=tol)
    if !istrue
        println(yellow!("backward  gradient: $∂L∂x"))
        println(yellow!("numerical gradient: $dLdx"))
    end
    if show
        println("backward  gradient: $∂L∂x")
        println("numerical gradient: $dLdx\n")
    end
    return istrue
end
