@testset "check gradient for MLP and Dense blocks" begin
    Random.seed!(UInt(time_ns()))
    T = Array{Float64}
    m = MLP([256, 128,120,100,80,60, 64], [relu,relu,relu!,leakyrelu,relu,relu]; type=T)
    @test checkgrad(m, Variable(randn(256, 62), type=T))
end
