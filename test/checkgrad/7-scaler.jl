@testset "check ScalePath op's gradient" begin
    # [1] prepare input data and its label
    input = randn(19,17,7);
    label = randn(19,17,7);
    x = Variable(input, type=Array{Float64});
    l = Variable(label, type=Array{Float64});

    # [2] prepare a learnabel scalar multiplier
    k = ScalePath(1.0, ndims=3, type=Array{Float64});
    fn(x) = MSELoss(forward(k, x), l);
    
    # [3] forward and get gradient by backpropagation
    @test checkgrad(fn, x)
end
