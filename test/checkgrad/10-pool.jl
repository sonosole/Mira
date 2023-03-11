@testset "check PoolLoss op's gradient" begin
    using Random
    Random.seed!(UInt(time_ns()))
    input = Variable(2randn(8,10,2), type=Array{Float64})
    LOSS1(x) = PoolLoss(softmax(x,dims=1), [[2,2,2,2,4],[0]], blank=1)
    @test checkgrad(LOSS1,  input, eps=1e-9)

    LOSS2(x) = PoolLoss(sigmoid(x), [[2,2,2,2,4],[3]])
    @test checkgrad(LOSS2,  input, eps=1e-9)
end
