export convfield
export tcnfield
export tcnlayers

"""
  convfield(KDS::Vector{NTuple{3,Dims{D}}}) where D

The input spatial width is `I`, then the output of the spatial width is:

          I - [D*(K-1) + 1]
      O = ───────────────── + 1
                  S

of which `D` is dialetion, `K` is kernel width, `S` is stride. vice versa
we got the input spatial width:

      I = (O - 1) * S + D*(K-1) + 1

when we have multi-layers, we can recurssively calculate from the top layer
to the bottom layer, assuming the top layer only has 1 receptive field.
# Example
      julia> kernel_dilation_stride = [
             ((3,4),(1,1),(2,2)),  # layer 1
             ((3,3),(1,1),(1,2)),  # layer 2
             ((4,2),(1,1),(2,1))   # layer 3
             ];

      julia> convfield(kernel_dilation_stride)
      (13, 12)
"""
function convfield(KDS::Vector{NTuple{3,Dims{D}}}) where D
    N = length(KDS) # layers in order
    KernelStride = Vector{NTuple{2,Dims{D}}}(undef, N)
    for i in 1:N
        kernel, dilation, stride = KDS[i]
        ekernel = ntuple(k -> dilation[k] * (kernel[k] - 1) + 1, D)
        KernelStride[i] = (ekernel, stride)
    end

    cfield = ntuple(D) do d
        i = 1
        for (kernel, stride) in reverse(KernelStride)
            i = (i-1) * stride[d] + kernel[d]
        end
        return i
    end
    return cfield
end


"""
  convfield(KS::Vector{NTuple{2,Dims{D}}}) where D
Suppose dilations are one.
# Example
      julia> kernel_stride = [
             ((3,4),(2,2)),  # layer 1
             ((3,3),(1,2)),  # layer 2
             ((4,2),(2,1))   # layer 3
             ];

      julia> convfield(kernel_stride)
      (13, 12)
"""
function convfield(KS::Vector{NTuple{2,Dims{D}}}) where D
    cfield = ntuple(D) do d
        i = 1
        for (kernel, stride) in reverse(KS)
            i = (i-1) * stride[d] + kernel[d]
        end
        return i
    end
    return cfield
end


"""
  convfield(KS::Vector{Dims{2}}, offset::Int=1)

For Conv1d conv field calculation. Suppose dilations are one.
# Example
      julia> kernel_stride = [
             (3,2),  # layer 1
             (3,1),  # layer 2
             (4,2)   # layer 3
             ];

      julia> convfield(kernel_stride)
      13
"""
function convfield(KS::Vector{Dims{2}}, offset::Int=1)
    i = offset
    for (kernel, stride) in reverse(KS)
        i = (i-1) * stride + kernel
    end
    return i
end



"""
    tcnfield(; kernel::Int, dilation::Int, layers::Int)

In TCN, also called Dialated Convolution Network, the kernel width is fixed for all
layers, the stride is 1 for all layers, and the dilation from the first layer to the
last layer is d⁰, d¹, d², … , dⁿ, since we already know `I = (O - 1) * S + D*(K-1) + 1`,
so we get the following:

    I₁   = O₁   + D₁  * (K-1)
    I₂   = O₂   + D₂  * (K-1),  I₂   = O₁
    I₃   = O₃   + D₃  * (K-1),  I₃   = O₂
            ⋯ ⋯ ⋯ ⋯ ⋯ ⋯
    Iₙ₋₂ = Oₙ₋₂ + Dₙ₋₂ * (K-1), Iₙ₋₂ = Oₙ₋₃
    Iₙ₋₁ = Oₙ₋₁ + Dₙ₋₁ * (K-1), Iₙ₋₁ = Oₙ₋₂
    Iₙ   = Oₙ   + Dₙ   * (K-1), Iₙ   = Oₙ₋₁

in which the top layer's receptive field `Oₙ = 1`, and `Dₙ = dⁿ⁻¹`, so the bottom input
width (i.e. receptive field) is

    I₁ = Oₙ + Dₙ * (K-1) + Dₙ₋₁ * (K-1) + ⋯ + D₁ * (K-1)
       = Oₙ + (K-1) * (d*Dₙ - D₁) / (d - 1)
       = 1 + (K-1) * (dⁿ - 1) / (d - 1)
"""
function tcnfield(; kernel::Int, dilation::Int, layers::Int)
    return (dilation^layers - 1) ÷ (dilation - 1) * (kernel - 1) + 1
end


"""
    tcnlayers(; kernel::Int, dilation::Int, field::Int)

Calculate the least layers needed, when given field, dilation and kernel width.
"""
function tcnlayers(; kernel::Int, dilation::Int, field::Int)
    t = (field - 1) / (kernel - 1) * (dilation - 1) + 1
    return ceil(Int, log(t) / log(dilation))
end
