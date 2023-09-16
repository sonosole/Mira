@testset "check MaxPool2d" begin
    #=
    x[1,1:3,1:6,1] =
    [1]  (4)  [7]  (10)  [13]  (16)
     2    5    8    11    14    17
    [3]  (6)  [9]  (12)  [15]  (18)

    [∙]-ed elements are in 1-st patch
    (∙)-ed elements are in 2-nd patch
    =#
    x = zeros(2,3,6,2); # 2 channels, 3×6 pixels, batchsize=2
    x[1,1:3,1:6,1] .= reshape(collect(1:18),3,6) .+ 0
    x[2,1:3,1:6,1] .= reshape(collect(1:18),3,6) .+ 1
    x[1,1:3,1:6,2] .= reshape(collect(1:18),3,6) .+ 2
    x[2,1:3,1:6,2] .= reshape(collect(1:18),3,6) .+ 3
    pool2d = Pool2d(maximum, kernel=(2,3), dilation=(2,2), stride=(1, 1))
    y = reshape(predict(pool2d, x), 2,4)
    @test sum(y .== [15 18 17 20;
                     16 19 18 21]) == 8
end


@testset "check AvgPool2d" begin
    #=
    x[1,1:3,1:6,1] =
    [1]  (4)  [7]  (10)  [13]  (16)
     2    5    8    11    14    17
    [3]  (6)  [9]  (12)  [15]  (18)

    [∙]-ed elements are in 1-st patch
    (∙)-ed elements are in 2-nd patch
    =#
    x = zeros(2,3,6,2); # 2 channels, 3×6 pixels, batchsize=2
    x[1,1:3,1:6,1] .= reshape(collect(1:18),3,6) .+ 0
    x[2,1:3,1:6,1] .= reshape(collect(1:18),3,6) .+ 1
    x[1,1:3,1:6,2] .= reshape(collect(1:18),3,6) .+ 2
    x[2,1:3,1:6,2] .= reshape(collect(1:18),3,6) .+ 3
    pool2d = Pool2d(mean, kernel=(2,3), dilation=(2,2), stride=(1, 1))
    y = reshape(predict(pool2d, x), 2,4)
    @test sum(y .== [8 11 10 13;
                     9 12 11 14]) == 8
end
