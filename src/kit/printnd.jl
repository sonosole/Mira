export printnd
export print1d
export print2d
export print3d
export print4d
export print5d
const IntOrRange = Union{Int, UnitRange{Int64}, StepRange{Int64, Int64}}


"""
    printnd(x::AbstractArray{T,D}, C::IntOrRange, B::IntOrRange) where {T,D}
Display spatial data of `x`
+ C is index or indices in Channel-Dim of x
+ B is index or indices in Samples-Dim of x
`print1d`, `print2d`, `print3d`, `print4d`, `print5d` are convinient functions for 1 to 5 dims data.
# Example
    julia> x = randn(5,3,4,2);
    julia> printnd(x,3,2)
    3×4 Matrix{Float64}:
     -1.70886    0.24668    -0.392457   0.00287569
     -0.761194  -0.0671505  -1.22395   -0.102049
      1.19511   -0.390474    0.997401   1.37079

    julia> printnd(x,3,1:2)
    3×4×2 Array{Float64, 3}:
    [:, :, 1] =
      0.496543   -0.695847   0.231459   -1.51813
      1.73864     1.4965    -0.0787232  -0.421759
     -0.0121892  -0.364314   0.361402   -1.32126

    [:, :, 2] =
      0.393683   0.510683   -1.2413     0.303262
      0.810592  -0.912774   -0.712823  -0.883501
     -0.242389   0.0823079  -0.542045   0.635964
"""
function printnd(x::AbstractArray{T,D}, C::IntOrRange, B::IntOrRange) where {T,D}
    sizex = size(x)
    spanx = CartesianIndices(ntuple(i -> 1:sizex[i+1], D-2))
    return display(x[C, spanx, B])
end


"""
    print1d(x::AbstractArray{T,3}, C::IntOrRange, B::IntOrRange) where T
Display spatial data of `x`
+ C is number of Channels of x
+ B is number of samples of x
# Example
    julia> x = randn(3,4,2);
    julia> print1d(x,3,2)
    4-element Vector{Float64}:
     1.6492889868957656
     1.0906245869909883
     1.9831031213092922
     1.0656825172757871

    julia> print1d(x,3,1:2)
    4×2 Matrix{Float64}:
     -1.54253    -0.306876
      0.0762228   0.587579
      1.16526     0.121794
     -0.958399    0.0459042
"""
function print1d(x::AbstractArray{T,3}, C::IntOrRange, B::IntOrRange) where T
    sizex = size(x)
    spanx = CartesianIndices(ntuple(i -> 1:sizex[i+1], 1))
    return display(x[C, spanx, B])
end


"""
    print2d(x::AbstractArray{T,4}, C::IntOrRange, B::IntOrRange) where T
Display spatial data of `x`
+ C is number of Channels of x
+ B is number of samples of x
# Example
    julia> x = randn(2,3,3,2);
    julia> print2d(x,3,2)
    3×3 Matrix{Float64}:
     -0.219888   2.10497    2.43549
      0.181104   0.300862   1.7673
      0.611012  -0.905083  -1.60214

    julia> print2d(x,1:2,1)
    2×3×3 Array{Float64, 3}:
    [:, :, 1] =
     -0.702468  -1.42604    -0.147491
     -1.03009   -0.0140063  -0.460441

    [:, :, 2] =
     -1.41441  -0.501106  0.199538
     -1.66479  -0.511232  0.891768

    [:, :, 3] =
      0.252448  -1.14865     -0.298948
     -0.460645   0.00317471  -0.605853
"""
function print2d(x::AbstractArray{T,4}, C::IntOrRange, B::IntOrRange) where T
    sizex = size(x)
    spanx = CartesianIndices(ntuple(i -> 1:sizex[i+1], 2))
    return display(x[C, spanx, B])
end


"""
    print3d(x::AbstractArray{T,5}, C::IntOrRange, B::IntOrRange) where T
Display spatial data of `x`
+ C is number of Channels of x
+ B is number of samples of x
# Example
    julia> x = randn(2,3,3,2,2);
    julia> print3d(x,1,1)
    3×3×2 Array{Float64, 3}:
    [:, :, 1] =
     -0.598021  -0.128303  1.53635
      0.278722   0.645246  1.3098
      0.606557   0.969783  0.0818521

    [:, :, 2] =
      0.296401  -0.017966  -0.655169
     -0.395186   0.775746   0.599534
      2.11601   -1.25374    1.33587
"""
function print3d(x::AbstractArray{T,5}, C::IntOrRange, B::IntOrRange) where T
    sizex = size(x)
    spanx = CartesianIndices(ntuple(i -> 1:sizex[i+1], 3))
    return display(x[C, spanx, B])
end


"""
    print4d(x::AbstractArray{T,6}, C::IntOrRange, B::IntOrRange) where T
Display spatial data of `x`
+ C is number of Channels of x
+ B is number of samples of x
"""
function print4d(x::AbstractArray{T,6}, C::IntOrRange, B::IntOrRange) where T
    sizex = size(x)
    spanx = CartesianIndices(ntuple(i -> 1:sizex[i+1], 4))
    return display(x[C, spanx, B])
end


"""
    print5d(x::AbstractArray{T,7}, C::IntOrRange, B::IntOrRange) where T
Display spatial data of `x`
+ C is number of Channels of x
+ B is number of samples of x
"""
function print5d(x::AbstractArray{T,7}, C::IntOrRange, B::IntOrRange) where T
    sizex = size(x)
    spanx = CartesianIndices(ntuple(i -> 1:sizex[i+1], 5))
    return display(x[C, spanx, B])
end
