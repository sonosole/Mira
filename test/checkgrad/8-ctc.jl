@testset "check CTCLoss op's gradient" begin
    SOFTMAX(x) = softmax(x, dims=1)

    # [1] creat model
    F = 32;   # featdim
    T = 64;   # timesteps
    B =  4;   # batchsize

    # [2] prepare input and its sequence label
    x = Variable(randn(F,T,B), type=Array{Float64});
    l = [[7,3,4], [4,5,6], [6,7], [0]];
    fn(x) = FRNNSoftmaxCTCLoss(SOFTMAX(x), l, blank=1, reduction="seqlen");

    # [3] checkgrad
    @test checkgrad(fn, x)
end
