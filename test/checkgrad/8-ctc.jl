@testset "check CTCLoss op's gradient" begin
    Random.seed!(UInt(time_ns()))
    SOFTMAX(x) = softmax(x, dims=1)

    # [1] creat model
    F = 32;   # featdim
    T = 64;   # timesteps
    B =  4;   # batchsize

    # [2] prepare input and its sequence label
    x = Variable(randn(F,T,B), type=Array{Float64});
    l = [[7,3,4], [4,5,6], [6,7], [0]];
    fn1(x) = FRNNSoftmaxCTCLoss(x, l, blank=1, reduction="seqlen");
    fn2(x) = FRNNSoftmaxFastCTCLoss(x, l, blank=1, reduction="seqlen");

    # [3] checkgrad
    @test checkgrad(fn1, x)
    @test checkgrad(fn2, x)
end
