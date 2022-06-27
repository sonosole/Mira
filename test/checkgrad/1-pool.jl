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

                fn(x) = MSELoss(pool(x, dims=DIMS), l);
                @test checkgrad(fn, x)
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

            fn(x) = MSELoss(pool(x; dims=DIMS), l);
            @test checkgrad(fn, x)
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

            fn(x) = MSELoss(pool(x; dims1=DIM1, dims2=DIM2), l)
            @test checkgrad(fn, x)
        end
    end
end
