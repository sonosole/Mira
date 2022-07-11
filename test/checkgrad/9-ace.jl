@testset "check ACELoss op's gradient" begin
    using Random
    Random.seed!(UInt(time_ns()))

    ACE(x) = ACELoss(softmax(x,dims=1), [[2,2,2,2,4],[0]], blank=1)
    @test checkgrad(ACE,  Variable(3*randn(8,10,2),type=Array{Float64}) )
end
