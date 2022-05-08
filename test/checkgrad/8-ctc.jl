@testset "check CTCLoss op's gradient" begin
    relux2y(x) = min2max(x, lower=-50.0, upper=50.0)
    SOFTMAX(x) = softmax(x, dims=1)

    mutable struct AModel
        blocks::Vector
        function AModel(featdims::Int, tones::Int)
            c1 = PlainConv1d(featdims, 256, 8, stride=3) # m[1]

            f1 = Linear(256, 256)          # m[2]
            f2 = Affine(256, 256)          # m[3]
            f3 = Dense(256, 256, relu)     # m[4]
            f4 = Dense(256, 256, sin)      # m[5]
            f5 = Dense(256, 256, sigmoid)  # m[6]
            f6 = Dense(256, 256, softplus) # m[7]
            f7 = Dense(256, 256, abs)      # m[8]

            chain = Chain(
                GRU(256, 256),                 # m[9][1]
                IndGRU(256, 256),              # m[9][2]
                LSTM(256, 256),                # m[9][3]
                IndLSTM(256, 256),             # m[9][4]
                RNN(256, 256, leakyrelu),           # m[9][5]
                IndRNN(256, 256, relu),   # m[9][6]
                PickyRNN(256, tones, relux2y)) # m[9][7]

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
    F = 32;   # featdim
    T = 64;  # timesteps
    B = 4;    # batchsize
    TYPE  = Array{Float64};
    MODEL = AModel(F, 32);
    model = clone(MODEL, type=TYPE);
    param = xparamsof(model);

    # [1] prepare input data and its label
    i = randn(F,T,B);
    x = Variable(i, type=TYPE);
    l = [[7 3 4], [4,5,6], [6,7], [8,9,15]];
    RE = "nil"
    KG = false
    BY = "dfs"

    # [2] forward and backward propagation
    y1 = forward(model, x);
    c1 = FRNNSoftmaxCTCLoss(y1, l, blank=1, weight=1.0, reduction=RE);
    backward(c1, by=BY, keepgraph=KG)
    GRAD1 = model[1].w.delta[1];
    zerograds!(param)

    # [3] with a samll change of a weight
    DELTA = 1e-7;
    model[1].w.value[1] += DELTA;

    # [4] forward and backward propagation
    y2 = forward(model, x);
    c2 = FRNNSoftmaxCTCLoss(y2, l, blank=1, weight=1.0, reduction=RE);
    backward(c2, by=BY, keepgraph=KG)
    GRAD2 = model[1].w.delta[1];
    zerograds!(param)

    GRAD = (GRAD1 + GRAD2)/2
    dLdW = (cost(c2) - cost(c1))/DELTA;   # numerical gradient
    err  = abs((dLdW-GRAD)/(GRAD+eps(Float64)))*100;  # relative error in %
    @test err < 1e-1
end
