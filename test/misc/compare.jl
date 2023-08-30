@testset "test logit x > y" begin
    x = Variable([5], keepsgrad=true)
    y = Variable([8], keepsgrad=false)
    for i in 1:10
        c = Loss(x > y)
        l = cost(c)
        if l > 0
            # println("$(x.value[1]) ≤ $(y.value[1])" |> yellow!)
        else
            @test x.value[1] > y.value[1]
            # println("$(x.value[1]) > $(y.value[1])" |> green!)
            break
        end
        backward(c)
        update!(x, 1.0)
        zerograds!(x)
    end
end


@testset "test logit x ≥ y" begin
    x = Variable([5], keepsgrad=true)
    y = Variable([8], keepsgrad=false)
    for i in 1:10
        c = Loss(x ≥ y)
        l = cost(c)
        if l > 0
            # println("$(x.value[1]) < $(y.value[1])" |> yellow!)
        else
            @test x.value[1] ≥ y.value[1]
            # println("$(x.value[1]) ≥ $(y.value[1])" |> green!)
            break
        end
        backward(c)
        update!(x, 1.0)
        zerograds!(x)
    end
end


@testset "test logit x < y" begin
    x = Variable([8], keepsgrad=true)
    y = Variable([5], keepsgrad=false)
    for i in 1:10
        c = Loss(x < y)
        l = cost(c)
        if l > 0
            # println("$(x.value[1]) ≥ $(y.value[1])" |> yellow!)
        else
            @test x.value[1] < y.value[1]
            # println("$(x.value[1]) < $(y.value[1])" |> green!)
            break
        end
        backward(c)
        update!(x, 1.0)
        zerograds!(x)
    end
end


@testset "test logit x ≤ y" begin
    x = Variable([8], keepsgrad=true)
    y = Variable([5], keepsgrad=false)
    for i in 1:10
        c = Loss(x ≤ y)
        l = cost(c)
        if l > 0
            # println("$(x.value[1]) > $(y.value[1])" |> yellow!)
        else
            @test x.value[1] ≤ y.value[1]
            # println("$(x.value[1]) ≤ $(y.value[1])" |> green!)
            break
        end
        backward(c)
        update!(x, 1.0)
        zerograds!(x)
    end
end


@testset "test logit 1 ≤ x ≤ 2" begin
    Random.seed!(UInt(time_ns()))
    x = Variable(3rand(4,8), keepsgrad=true)
    # display(x.value)
    for _ in 1:20
        y = Loss(1 ≤ x) + Loss(x ≤ 2)
        println(cost(y))
        if cost(y) == 0
            @test true
            # display(x.value)
            break
        end
        backward(y)
        update!(x, 0.5)
        zerograds!(x)
    end
end
