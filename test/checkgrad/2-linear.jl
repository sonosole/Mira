@testset "check gradient for Linear block " begin
    using Random
    Random.seed!(UInt(time_ns()))
    T = Array{Float64}
    m = Linear(256, 64; type=T)
    x = Variable(randn(256, 62), type=T);
    @test checkgrad(m, x)
end
