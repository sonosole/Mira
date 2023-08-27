@testset "check gradient for Conv1d block" begin
    using Random
    Random.seed!(UInt(time_ns()))
    TYPE = Array{Float64};
    ichannels = 2
    ochannels = 3
    timeSteps = 64
    batchsize = 2
    x = Variable(rand(ichannels, timeSteps, batchsize); type=TYPE)

    for (K,D,S) in [(3,3,5), (3,2,5), (2,5,4)], P in ["valid", 10, (5,9)]
        c1 = Conv1d(ichannels, 4, kernel=K, dilation=D, stride=S, padding=P, type=TYPE)
        c2 = Conv1d(4, ochannels, kernel=K, dilation=D, stride=S, padding=P, type=TYPE)
        @test checkgrad(Chain(c1,c2), x)
    end

    for K in [3], D in [3], S in [8], P in [(5,10)],
        M in ["zeros", "constant", "repeat", "reflect", "symmetric", "circular"]
        c1 = Conv1d(ichannels, 4, kernel=K, dilation=D, stride=S, padding=P, padmode=M, type=TYPE)
        c2 = Conv1d(4, ochannels, kernel=K, dilation=D, stride=S, padding=P, padmode=M, type=TYPE)
        @test checkgrad(Chain(c1,c2), x)
    end

    for (K,D,S) in [(3,3,1), (3,2,1), (2,5,1)], P in ["same"]
        c1 = Conv1d(ichannels, 4, kernel=K, dilation=D, stride=S, padding=P, type=TYPE)
        c2 = Conv1d(4, ochannels, kernel=K, dilation=D, stride=S, padding=P, type=TYPE)
        @test checkgrad(Chain(c1,c2), x)
    end
end


@testset "check gradient for Conv2d block" begin
    using Random
    Random.seed!(UInt(time_ns()))
    TYPE = Array{Float64};
    Ci = 2
    Co = 3
    H = 64
    W = 39
    B = 2
    x = Variable(rand(Ci, H, W, B); type=TYPE)

    for (K,D,S) in [((3,2),(3,3),(5,4)), ((3,4),(2,2),(5,5)), ((2,2),(5,3),(4,1))], P in ["valid", 10, ((5,9), (2,4))]
        c1 = Conv2d(Ci, 4, kernel=K, dilation=D, stride=S, padding=P, type=TYPE)
        c2 = Conv2d(4, Co, kernel=K, dilation=D, stride=S, padding=P, type=TYPE)
        @test checkgrad(Chain(c1,c2), x)
    end

    for K in [(3,2)], D in [(3,1)], S in [(8,4)], P in [(5,10),(3,2)],
        M in ["zeros", "constant", "repeat", "reflect", "symmetric", "circular"]
        c1 = Conv2d(Ci, 4, kernel=K, dilation=D, stride=S, padding=P, padmode=M, type=TYPE)
        c2 = Conv2d(4, Co, kernel=K, dilation=D, stride=S, padding=P, padmode=M, type=TYPE)
        @test checkgrad(Chain(c1,c2), x)
    end

    for (K,D) in [((3,2),(3,3)), ((3,4),(2,2)), ((2,2),(5,3))], P in ["same"]
        c1 = Conv2d(Ci, 4, kernel=K, dilation=D, padding=P, type=TYPE)
        c2 = Conv2d(4, Co, kernel=K, dilation=D, padding=P, type=TYPE)
        @test checkgrad(Chain(c1,c2), x)
    end
end
