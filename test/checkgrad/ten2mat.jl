@testset "test ten2mat's result" begin
    #=                  0. 0  0. 0   0. 0  0
     1  4  7  10        0  0  0  0   0  0  0
     2  5  8  11   →    0  1  4  7  10  0  0
     3  6  9  12        0. 2  5. 8  11. 0  0
                        0  3  6  9  12  0  0
                        0  0  0  0   0  0  0
                        0  0  0  0   0  0  0
    =#
    p = ((2,2),(1,2));
    k = (2,3);
    d = (3,2);
    s = (1,2);

    xten = reshape(collect(1:12), 1,3,4,1);
    xmat = ten2mat(xten, p, k, d, s, padconst, 0);
    mat  = [0   0   0   0   0   0   4   5;
            0   0   0   0   5   6   0   0;
            0   0   4   5   0   0  10  11;
            5   6   0   0  11  12   0   0;
            0   0  10  11   0   0   0   0;
           11  12   0   0   0   0   0   0]; # 6×8
    @test sum(xmat - mat) == 0
end


@testset "test ten2mat's grad with no parallizing" begin
    p = ((2,2),(1,2));
    k = (2,3);
    d = (3,2);
    s = (1,2);

    mat = [0   0   0   0   0   0   4   5;
           0   0   0   0   5   6   0   0;
           0   0   4   5   0   0  10  11;
           5   6   0   0  11  12   0   0;
           0   0  10  11   0   0   0   0;
          11  12   0   0   0   0   0   0]; # 6×8

    grad = [0  2  0  2;
            0  4  0  4;
            0  2  0  2]; # 3×4

    xten = Variable(reshape(collect(1:12), 1,3,4,1),keepsgrad=true);
    xmat = ten2mat(xten, p, k, d, s, padconst, 0);
    @test sum(xmat.value - mat) == 0
    backward(xmat)
    @test sum(reshape(xten.delta, 3,4) - grad) == 0
end


@testset "test ten2mat's grad with parallizing" begin
    p = ((0,0),(0,0));
    k = (2,2);
    d = (1,1);
    s = (3,3);

    mat  = [1  3  13  15;
            2  4  14  16;
            5  7  17  19;
            6  8  18  20]; # 4×4

    grad = [1  1  0  1  1  0;
            1  1  0  1  1  0;
            0  0  0  0  0  0;
            1  1  0  1  1  0;
            1  1  0  1  1  0]; # 5×6

    ten  = reshape([1 5  9 13 17 21;
                    2 6 10 14 18 22;
                    0 0  0  0  0  0;
                    3 7 11 15 19 23;
                    4 8 12 16 20 24], (1,5,6,1));

    xten = Variable(ten, keepsgrad=true);
    xmat = ten2mat(xten, p, k, d, s, padconst, 0);
    @test sum(xmat.value - mat) == 0
    backward(xmat)
    @test sum(reshape(xten.delta, 5,6) - grad) == 0
end
