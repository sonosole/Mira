# assign a pooling function
poolfn(x) = powerpool(x, 0.5, dims=1)
# assign label. 1 for positive, 0 for negative
label = ones(Float32, 1, 1); # positive value

# creat two peaks
t = -20:0.5:30;
a = @. 7exp(-((t+8)/4)^2) + 3*exp(-((t-9)/6)^2) - 4;
b = reshape(a, length(a),1);
𝟙 = ones(size(x));

# wrap two peaks
x = Variable(b);
needsgrad(x);

# plot training process
for i = 1:512
    # forward
    y = sigmoid(x)
    z = poolfn(y)
    l = FocalBCELoss(z, label)

    # backward and update
    backward(l)
    update!(x, 2.0)
    zerograds!(x)

    # init states
    plot(t, sigmoid(a), label="original")
    # updated states
    plot!(t, y.value, label="updated")
    # aggresive value
    plot!(t, z.value ./ 2.0 .* 𝟙, ylims=(0.0,1.0), label="half pooled value")
    # keep plot alive
    gui()
end
