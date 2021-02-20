using LinearAlgebra, Statistics
using JLD2, BSON
using Flux, CUDA
using Plots
using Flux: Data.DataLoader, update!, onehot, onehotbatch, chunk, batchseq
using Zygote
using Base.Iterators:partition, product

device!(0) # CUDA device 1, Gerlach
gr()
theme(:juno)

BSON.@load "saved_models/modelv4_adam_399ep.bson" net
JLD2.@load "data/moving_forest_v2.jld2" ts as
# JLD2.@load "data/moving_forest_8k_training.jld2" train_data

CUDA.allowscalar(false)
includet("utils.jl")

##
const movements = Dict(
    "left"  => [-1, 0],
    "right" => [1,  0],
    "up"    => [0,  1],
    "down"  => [0, -1],
    )

const mov_inds = Dict(
    "left"  => 1,
    "right" => 2,
    "up"    => 3,
    "down"  => 4,
    )

function prop_err(net, x)
    net.W' * x
end

##

function load_data(filename)
    @load filename ts as
    action_labels = map(x -> Float32.(Flux.onehotbatch(x, 1:4)), as)
    xs = map(x -> reshape(Float32.(x), 256, 40), ts)
    xs, action_labels
end

@inline function get_r(net, x)
    r = prop_err(net, x)
    rhat = ISTA(x, r, net, η=0.01f0, λ=0.001f0, target=0.001f0)
    # normalize
    mapslices(normalize, r, dims=1)
end

xs, as = load_data("data/moving_forest_v3_8k.jld2")
# r = get_r(net, xs[4])

@inline function prepare_data(filename, batchsize)
    xs, as = load_data(filename)
    N = length(xs)
    # sparse vectors
    rs = [get_r(net, xs[i]) for i in 1:N]
    # stack action
    stacked_xs = map((x, y) -> [x ; y], rs, as)
    labels = push!(rs[2:end], zeros(Float32, size(rs[1])))

    xy = @. convert(Array, VectorOfArray(([stacked_xs, labels])))
    f = @. Int(floor(N / batchsize))
    xy =  map(x -> reshape(x, size(x)[1:2]..., batchsize, f), xy)
    # return
    map(x -> [[x[:,k,:,i] for k in 1:40] for i in 1:f], xy)
end

##
# xtrain, ytrain = prepare_data("data/moving_forest_v3_8k.jld2", 25)

@load "data/mov_forest_training_data_v3_8k.jld2" xtrain ytrain

##
batchsize = 25
# xtest, ytest = prepare_data("data/moving_forest_v2.jld2", batchsize)
0.8 * length(xtrain)

x_train = xtrain[1:256]
x_test = xtrain[256:end]

y_train = ytrain[1:256]
y_test = ytrain[256:end]

##
@inline function loss_fn(x, y)
    # Flux.reset!(Rnet)
    r̂ = Rnet.(x)
    l = sum(Flux.mse.(r̂, y))
    l
end

@inline Zygote.@nograd function eval_model(f, xtest, ytest)
    l = f.(xtest, ytest)
    mean(l)
end

function train_model!(xs, ys, opt, lossfn, ps)
    for (i, (x, y)) in enumerate(zip(xs, ys))
        x, y = x |> gpu, y |> gpu
        g = gradient(() -> lossfn(x, y), ps)
        update!(opt, ps, g)
        # Flux.reset!(Rnet)
    end
end

##
println("Meow")
##

optimizers = [ADAM, Descent]
lrs = [0.001]
# acts = [tanh, σ, relu]
Rs = [GRU, RNN]
Ns = [64, 128,]

options = [Rs, Ns, optimizers, lrs]

xtrain = x_train |> gpu
ytrain = y_train |> gpu
xtest, ytest = x_test |> gpu, y_test |> gpu
xt, yt = xtest[1:10], ytest[1:10]

train_data = [xtrain, ytrain, xt, yt]
opts = [(RNN,), (32,), (ADAM,), (0.01,)]

sparse_net = net
norm1(x) = mean(sum(abs, x, dims=1))
##

function try_model(data, options, epochs)
    xtrain, ytrain, xt, yt = data
    models, r̄ = [], []
    for config in product(options...)
        Rmodel, N, optimizer, lr = config
        @info "Creating model with $Rmodel, $N units and $((optimizer, lr))"
        # Model definition
        Rnet = Chain(
        Rmodel(100 + 4, N),
        Dense(N, 100),
        # x -> tanh.(x),
        ) |> gpu

        opt = optimizer(lr)
        ps = Flux.params(Rnet)

        loss(x, y) = begin
           out = Rnet.(x)
           sum(Flux.mse.(out, y) + 0.2f0 * map(norm1, out))
        end

        for epoch in 1:epochs
            Flux.train!(
            loss,
            ps,
            zip(xtrain, ytrain),
            opt, )
            val_loss = eval_model(loss, xt, yt)
            println("Epoch: $epoch, val loss: $(round(val_loss, digits=4))")
        end
        out = Rnet.(xt[8]) |> cpu
        r̂ = hcat([out[k][:,1] for k in 1:40]...) |> cpu
        Rnet = cpu(Rnet)
        push!(models, Rnet)
        push!(r̄, r̂)
    end
    Dict("models" => models, "r̄" => r̄)
end

##

model_dict = try_model(train_data, options, 100)

##

function plot_rs(ind)
    l = @layout [a ; b]
    tmp = model_dict["r̄"][ind]
    f1 = heatmap(tmp)
    title!("$(perms[ind]), $ind")
    f2 = heatmap(hcat([y_test[8][k][:,1] for k in 1:40]...))
    plot(f1, f2, layout=l)
end

perms = collect(product(options...))[:]

begin
    ind = 12
    plot_rs(ind)
end

pred = mapslices(sparse_net, model_dict["r̄"][ind], dims=1)
quick_anim(permutedims(reshape(pred, 16, 16, 40), [3, 2, 1]),
    savestring="$(join(perms[ind])).gif")
