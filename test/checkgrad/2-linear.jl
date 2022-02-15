@testset "check gradient for Linear block " begin
    # [1] prepare input data and its label
    T = Array{Float64}
    x = randn(256, 62)
    l = rand(64, 62)
    l = l ./ sum(l,dims=1)
    m = Linear(256, 64; type=T)

    x = Variable(x; type=T);
    l = Variable(l; type=T);

    # [2] forward and backward propagation
    outs = forward(m, x)
    LOSS1 = mse(outs, l)
    COST1 = loss(LOSS1)
    backward(COST1)
    GRAD = m.w.delta[1]

    # [3] with a samll change of a weight
    DELTA = 1e-6
    m.w.value[1] += DELTA

    # [4] forward and backward propagation
    outs = forward(m, x)
    LOSS2 = mse(outs, l)
    COST2 = loss(LOSS2)
    backward(COST2)

    # [5] check if the auto-grad is true or not
    dLdW = (ᵛ(COST2)[1] - ᵛ(COST1)[1])/DELTA;         # numerical gradient
    err  = abs((dLdW-GRAD)/(GRAD+eps(Float64)))*100;  # relative error in %
    @test err < 1e-1
end
