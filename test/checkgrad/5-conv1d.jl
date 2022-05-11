# The precision of forward-mode-differiation is accurate at Float64 but not at Float32/16.
# Anyway the backward-mode-differentiation is accurate at Float64/32/16.
# Just check at Float64/32/16 while keeping the same random seed.
# For the below example the gradients for CPU/GPU on my device are
# Float64 is 100.724249419
# Float32 is 100.72425
# Float16 is 100.75
# ochannels = 1

@testset "check gradient for PlainConv1d block" begin
    using Random
    Random.seed!(UInt(time_ns()))
    TYPE = Array{Float64};

    ichannels = 1
    ochannels = 2
    c1 = PlainConv1d(ichannels,4,3; type=TYPE)
    c2 = PlainConv1d(4,ochannels,2; type=TYPE)

    timeSteps = 128
    batchsize = 32
    x = Variable(rand(ichannels,timeSteps,batchsize); type=TYPE)
    @test checkgrad(Chain(c1,c2), x)
end
