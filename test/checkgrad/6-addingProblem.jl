@testset "check gradient for IndRNN block in adding problem" begin
    function addingProblemData(T::Int)
        #= 两个长度为 T 的序列组成输入：
        第一个序列在 (0,1)范围内均匀采样，
        第二个序列的前两个为 1，其余都为 0.
        例如：
        x1 = 0.90  0.93  0.90  0.56  0.46
        x2 = 1.0   1.0   0.0   0.0   0.0
        而输出为 sum(x1 .* x2) = 0.90 + 0.93
        =#
        @assert (T>1) "The sequence length should lager than 1"
        x1 = rand(1,T)./3
        x2 = zeros(1,T)
        x2[1] = 1.0
        x2[2] = 1.0
        y = sum(x1 .* x2)
        return [x1;x2],[y]
    end

    Random.seed!(UInt(time_ns()))
    TYPE = Array{Float64};

    # [0] prepare model
    model = Chain(
        IndRNN(2, 64, leakyrelu; type=TYPE),
        IndGRU(64, 64; type=TYPE),
        IndLSTM(64, 64; type=TYPE),
        RNN(64, 64, cos;  type=TYPE),
        LSTM(64, 64; type=TYPE),
        GRU(64, 64; type=TYPE),
        Dense(64,1, relu; type=TYPE)
    )

    # [1] prepare input data and its label
    T = 150
    x, s = addingProblemData(T)

    # [2] forward and backward propagation
    resethidden(model)
    for t = 1:T-1
        tmp = forward(model, Variable( reshape(x[:,t], 2,1); type=TYPE) );
    end
    y = forward(model, Variable( reshape(x[:,T], 2,1); type=TYPE) );
    COST1 = MSELoss(y, Variable( reshape(s,1,1); type=TYPE) );
    backward(COST1, partial=true, keepgraph=true)
    GRAD = model[1].w.delta[1]

    # [3] with a samll change of a weight
    DELTA = 1e-5
    model[1].w.value[1] += DELTA

    # [4] forward and backward propagation again
    resethidden(model)
    for t = 1:T-1
        tmp = forward(model, Variable( reshape(x[:,t], 2,1); type=TYPE) )
    end
    y = forward(model, Variable( reshape(x[:,T], 2,1); type=TYPE) )
    COST2 = MSELoss(y, Variable( reshape(s,1,1); type=TYPE) )
    backward(COST2, partial=true, keepgraph=true)

    # [5] check if the auto-grad is true or not
    dLdW = (cost(COST2) - cost(COST1))/DELTA
    err  = abs((dLdW-GRAD)/(GRAD+eps(Float64)))*100;  # relative error in %
    @test err < 0.1
end
