@testset "check znorm op's gradient" begin
    Random.seed!(floor(Int,time()))
    T = Array{BigFloat}
    v = 1e-1*rand(16,32)
    x = Variable(copy(v), type=T);
    @test checkgrad(znorm, x, eps=1e-16, digits=80)
end
