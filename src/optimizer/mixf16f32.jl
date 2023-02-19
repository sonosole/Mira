

"""
# float16 mix float32 process
+ [1] forward：
Y₁₆ = W₁₆ * X₁₆

+ [2] backward：
dX₁₆ = Wᵀ₁₆ * dY₁₆
dW₁₆ = dY₁₆ * Xᵀ₁₆

+ [3] update：
W₃₂ += - lr * dW₁₆

+ [4] sync to float16：
W₁₆ = tofloat32(W₃₂)
"""
mutable struct MixPrecision{Opt}
    optimizer :: Opt
    backup :: XVariables
    function name(opt::Opt; dtype=Array{Float32}) where Opt <: Optimizer
        num = length(opt.xparams)
        W₃₂ = VecXVariable(num)
        for i = 1:num
            c , θ = xparams[i]
            W₃₂[i] = (c, clone(θ, dtype=dtype))
        end
        new{opt}(opt, W₃₂)
    end
end


function update!(mixpre::MixPrecision,
                 clipfn::Function=LPInfNormClip,
                 clipvalue::Real=10.0,
                 applyL1::Function=decay_by_L₁,
                 applyL2::Function=decay_by_L₂)
    # update in Optimizer
    update!(mixpre.optimizer,
            clipfn=clipfn,
            clipvalue=clipvalue,
            applyL1=applyL1,
            applyL2=applyL2)
    # sync W₁₆ with W₃₂
    for (W₁₆, W₃₂) in zip(mixpre.optimizer.xparams, backup)
        W₁₆ .= W₃₂
    end
end


function zerograds!(mixpre::MixPrecision)
    Threads.@threads for xvar in mixpre.optimizer.xparams
        c , x = xvar
        if !isnothing(δ(x))
            δ(x) .= 0.0
        end
    end
end
