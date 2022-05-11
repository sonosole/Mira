@testset "check gradient for Chain" begin
    # [1] prepare input data and its label
    using Random
    Random.seed!(UInt(time_ns()))

    TYPE = Array{Float64}
    x = randn(256, 62)
    x = Variable(x; type=TYPE)

    blocks = [
        Dense(256,128,tanh; type=TYPE),
        Dense(128,128,cos;  type=TYPE),
        Dense(128,128,sin;  type=TYPE),
        Dense(128,128,cos!; type=TYPE),
        Dense(128,128,sin!; type=TYPE),
        Dense(128, 64,relu; type=TYPE),
        Maxout(64, 64; k=3, type=TYPE),
        Residual(
            Dense(64,32,sin;type=TYPE),
            Linear(32,64,   type=TYPE)
        )
    ]

    model = Chain(blocks)
    @test checkgrad(model, x)
end
