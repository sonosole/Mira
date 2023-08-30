@testset "check gradient for 1-D Paddings" begin
    Random.seed!(UInt(time_ns()))
    TYPE = Array{Float64};
    ichannels = 2
    timeSteps = 32
    batchsize = 2
    x = Variable(rand(ichannels, timeSteps, batchsize); type=TYPE)

    for padfn in [padconst, padrepeat, padreflect, padsymmetric, padcircular]
        fn(a) = padfn(a, ((0, 0), (5, 10), (0, 0)))
        @test checkgrad(fn, x)
    end
end


@testset "check gradient for 2-D Paddings" begin
    Random.seed!(UInt(time_ns()))
    TYPE = Array{Float64};
    C = 2
    H = 16
    W = 16
    B = 2
    x = Variable(rand(C, H, W, B); type=TYPE)

    for padfn in [padconst, padrepeat, padreflect, padsymmetric, padcircular]
        fn(a) = padfn(a, ((0, 0), (5, 10),(6, 7), (0, 0)))
        @test checkgrad(fn, x)
    end
end
