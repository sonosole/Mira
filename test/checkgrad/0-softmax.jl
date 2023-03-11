@testset "check softmax & polymax op's gradient" begin
    using Random
    Random.seed!(UInt(time_ns()))
    TYPE = Array{Float64};
    inputdims = 64;
    timeSteps = 16;
    batchsize = 32;
    x = Variable(0.1randn(inputdims, timeSteps, batchsize), type=TYPE);
    l = Variable(0.1randn(inputdims, timeSteps, batchsize), type=TYPE);

    for d in [1 2 3 (1,2) (2,3) (1,3) (1,2,3)]
        @testset "check softmax op's gradient at dim = $d" begin
            fn(x) = CrossEntropyLoss(softmax(x, dims=d), l)
            @test checkgrad(fn, x)
        end
    end

    for d in [1 2 3 (1,2) (2,3) (1,3) (1,2,3)]
        @testset "check polymax op's gradient at dim = $d" begin
            fn(x) = CrossEntropyLoss(polymax(x, 2, dims=d), l)
            @test checkgrad(fn, x)
        end
    end
end
