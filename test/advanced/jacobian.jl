@testset "check jacobian matrix" begin

    x = Variable(reshape([1 2 3; 4 5 6], 2,3))
    w = Variable(reshape([-1.0, 2.0]   , 1,2))
    y = w * x

    @test jacobian(y, w) ≈ [1.0  4.0
                            2.0  5.0
                            3.0  6.0]

    @test jacobian(y, x) ≈ [-1.0  2.0   0.0  0.0   0.0  0.0
                             0.0  0.0  -1.0  2.0   0.0  0.0
                             0.0  0.0   0.0  0.0  -1.0  2.0]
end
