@testset "check gradient for PlainDepthConv1d block" begin
    Random.seed!(UInt(time_ns()))
    TYPE = Array{Float64}
    CHANNEL = 8
    KERNEL  = 6
    STRIDE  = 1
    BLOCK   = PlainDepthConv1d(CHANNEL, kernel=KERNEL, stride=STRIDE, type=TYPE)

    timeSteps = 256
    batchsize = 32
    x = Variable(rand(CHANNEL, timeSteps, batchsize); type=TYPE)
    @test checkgrad(BLOCK, x)

    fn(x) = forward(BLOCK, x)
    @test checkgrad(fn, x, eps=1e-7, tol=1.0)
end
