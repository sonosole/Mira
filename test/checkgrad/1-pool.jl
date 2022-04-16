using Random
Random.seed!(UInt(time_ns()))

@testset "check pooling op's gradient at single dim" begin
    for d in [1 2 3]
        for pool in [maximum minimum sum mean linearpool exppool]
            @testset "check $pool op's gradient at dim = $d" begin
                DIMS = d
                TYPE = Array{Float64};
                # [1] prepare input data and its label
                inputdims = 64;
                timeSteps = 16;
                batchsize = 32;
                x = Variable(rand(inputdims, timeSteps, batchsize); type=TYPE, keepsgrad=true);
                if d==1;l = Variable(rand(1, timeSteps, batchsize); type=TYPE);end
                if d==2;l = Variable(rand(inputdims, 1, batchsize); type=TYPE);end
                if d==3;l = Variable(rand(inputdims, timeSteps, 1); type=TYPE);end

                # [2] forward and backward propagation
                COST1 = mseLoss(pool(x; dims=DIMS), l);
                backward(COST1);

                # [3] with a samll change of a weight
                GRAD = x.delta[1];
                DELTA = 1e-6;
                x.value[1] += DELTA;

                # [4] forward and backward propagation
                COST2 = mseLoss(pool(x; dims=DIMS), l);
                backward(COST2);

                # [5] check if the auto-grad is true or not
                dLdW = (cost(COST2) - cost(COST1))/DELTA;         # numerical gradient
                err  = abs((dLdW-GRAD)/(GRAD+eps(Float32)))*100;  # relative error in %
                @test err < 1e-1
            end
        end
    end
end


@testset "check pooling op's gradient at mutiple dims" begin
    for pool in [maximum minimum sum mean linearpool exppool]
        @testset "check $pool op's gradient" begin
            DIMS = (1,2)
            TYPE = Array{Float64};

            # [1] prepare input data and its label
            inputdims = 64;
            timeSteps = 16;
            batchsize = 32;
            x = Variable(rand(inputdims, timeSteps, batchsize); type=TYPE,keepsgrad=true);
            l = Variable(rand(1,         1,         batchsize); type=TYPE);

            # [2] forward and backward propagation
            COST1 = mseLoss(pool(x; dims=DIMS), l);
            backward(COST1);

            # [3] with a samll change of a weight
            GRAD = x.delta[1];
            DELTA = 1e-6;
            x.value[1] += DELTA;

            # [4] forward and backward propagation with a samll change of a weight
            COST2 = mseLoss(pool(x; dims=DIMS), l);
            backward(COST2);

            # [5] check if the auto-grad is true or not
            dLdW = (cost(COST2) - cost(COST1))/DELTA;         # numerical gradient
            err  = abs((dLdW-GRAD)/(GRAD+eps(Float64)))*100;  # relative error in %
            @test err < 1e-1
        end
    end
end


@testset "check maxmin and minmax op's gradient at mutiple dims" begin
    for pool in [maxmin minmax]
        @testset "check $pool op's gradient" begin
            DIM1 = 1
            DIM2 = 2
            TYPE = Array{Float64};

            # [1] prepare input data and its label
            inputdims = 64;
            timeSteps = 16;
            batchsize = 32;
            x = Variable(rand(inputdims, timeSteps, batchsize); type=TYPE,keepsgrad=true);
            l = Variable(rand(1,         1,         batchsize); type=TYPE);

            # [2] forward and backward propagation
            COST1 = mseLoss(pool(x; dims1=DIM1, dims2=DIM2), l)
            backward(COST1);
            GRAD = x.delta[1];

            # [3] with a samll change of a weight
            DELTA = 1e-6;
            x.value[1] += DELTA;

            # [4] forward and backward propagation with a samll change of a weight
            COST2 = mseLoss(pool(x; dims1=DIM1, dims2=DIM2), l)
            backward(COST2);

            # [5] check if the auto-grad is true or not
            dLdW = (cost(COST2) - cost(COST1))/DELTA;         # numerical gradient
            err  = abs((dLdW-GRAD)/(GRAD+eps(Float64)))*100;  # relative error in %
            @test err < 1e-1
        end
    end
end
