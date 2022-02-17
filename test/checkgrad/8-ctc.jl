@testset "check CTCLoss op's gradient" begin
    relux2y(x) = min2max(x, lower=-50.0, upper=50.0)
    SOFTMAX(x) = softmax(x, dims=1)

    mutable struct AModel
        blocks::Vector
        function AModel(featdims::Int, tones::Int)
            c1 = PlainConv1d(featdims, 512, 8, stride=3) # m[1]

            f1 = Dense(512, 512, relu!) # m[2]
            f2 = Dense(512, 512, relu!) # m[3]
            f3 = Dense(512, 512, relu!) # m[4]
            f4 = Dense(512, 512, relu!) # m[5]
            f5 = Dense(512, 512, relu!) # m[6]
            f6 = Dense(512, 512, relu!) # m[7]
            f7 = Dense(512, 512, relu!) # m[8]

            chain = Chain(
               RNN(512, 512, relu!),     # m[9][1]
               RNN(512, 512, relu!),     # m[9][2]
            IndRNN(512, tones, relux2y)) # m[9][3]

            new([c1, f1, f2, f3, f4, f5, f6, f7, chain])
        end
        function AModel(nblocks::Int)
            new(Vector(undef,nblocks))
        end
    end

    @extend(AModel, blocks)

    function Mira.clone(this::AModel; type::Type=Array{Float32})
        nblocks = length(this.blocks)
        cloned  = AModel(nblocks)
        for i = 1:nblocks
            cloned[i] = clone(this[i], type=type)
        end
        return cloned
    end

    # define model's forward calculation
    function Mira.forward(m::AModel, x::Variable)
        # --- conv1d layer
        x = relu!(forward(m[1], x))
        C,T,B = size(x);
        x = reshape(x, (C, T*B))  # 3D --> 2D

        # --- dense layers
        for i = 2:8
            x = forward(m[i], x)
        end
        C,TB = size(x);
        x = reshape(x, (C,T,B))  # 2D --> 3D

        # rnn block
        y = PackedSeqForward(m[9], x)
        return y
    end

    # [0] model
    F = 64;   # featdim
    T = 1250; # timesteps
    B = 4;    # batchsize
    TYPE  = Array{Float64};
    MODEL = AModel(F, 32);
    model = clone(MODEL, type=TYPE);
    param = xparamsof(model);

    # [1] prepare input data and its label
    i = randn(F,T,B);
    x = Variable(i, type=TYPE);
    l = [[2 3 4], [4,5,6], [6,7], [8,9,15]];
    RE = "nil"
    KG = false

    # [2] forward and backward propagation
    y1 = forward(model, x);
    c1 = CRNN_Batch_CTC_With_Softmax(y1, l, blank=1, weight=1.0, reduction=RE);
    backward(c1, by="dfs", keepgraph=KG)
    GRAD = model[1].w.delta[1];

    # [3] with a samll change of a weight
    DELTA = 1e-7;
    model[1].w.value[1] += DELTA;

    # [4] forward and backward propagation
    y2 = forward(model, x);
    c2 = CRNN_Batch_CTC_With_Softmax(y2, l, blank=1, weight=1.0, reduction=RE);
    backward(c2, by="dfs", keepgraph=KG)
    zerograds!(param)

    dLdW = (cost(c2) - cost(c1))/DELTA;   # numerical gradient
    err  = abs((dLdW-GRAD)/(GRAD+eps(Float64)))*100;  # relative error in %
    println("\nBWD ",GRAD);
    println("SHU ",dLdW);
    println("ERR ",trunc(err,digits=3), " %");
    @test err < 1e-1
end
