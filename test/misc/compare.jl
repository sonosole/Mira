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
