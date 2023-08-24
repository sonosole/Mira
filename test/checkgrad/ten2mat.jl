
@testset "test ten2mat's result" begin
    p = ((2,2),(1,2));
    k = (2,3);
    d = (3,2);
    s = (1,2);

    # Array type
    xten = reshape(collect(1:12), 1,3,4,1);
    xmat = ten2mat(xten, p, k, d, s, padconst, 0);
    mat  = [0   0   0   0   0   0   4   5;
            0   0   0   0   5   6   0   0;
            0   0   4   5   0   0  10  11;
            5   6   0   0  11  12   0   0;
            0   0  10  11   0   0   0   0;
           11  12   0   0   0   0   0   0];
    @test sum(xmat - mat) == 0
end


@testset "test ten2mat's grad" begin
    p = ((2,2),(1,2));
    k = (2,3);
    d = (3,2);
    s = (1,2);

    mat  = [0   0   0   0   0   0   4   5;
            0   0   0   0   5   6   0   0;
            0   0   4   5   0   0  10  11;
            5   6   0   0  11  12   0   0;
            0   0  10  11   0   0   0   0;
           11  12   0   0   0   0   0   0];

    delta = [0  2  0  2;
             0  4  0  4;
             0  2  0  2];

    xten = Variable(reshape(collect(1:12), 1,3,4,1),keepsgrad=true);
    xmat = ten2mat(xten, p, k, d, s, padconst, 0);
    @test sum(xmat.value - mat) == 0
    backward(xmat)
    @test sum(reshape(xten.delta, 3,4) - delta) == 0
end
