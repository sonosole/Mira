export convfield

"""
  convfield(KDS::Vector{NTuple{3,Dims{D}}}) where D
# example
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
  convfield(KDS::Vector{NTuple{2,Dims{D}}}) where D
# example
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
  convfield(KS::Vector{NTuple{2,Int}})

For Conv1d conv field calculation
# example
  julia> kernel_stride = [
         (3,2),  # layer 1
         (3,1),  # layer 2
         (4,2)   # layer 3
         ];

  julia> convfield(kernel_stride)
  13
"""
function convfield(KS::Vector{NTuple{2,Int}})
    i = 1
    for (kernel, stride) in reverse(KS)
        i = (i-1) * stride + kernel
    end
    return i
end
