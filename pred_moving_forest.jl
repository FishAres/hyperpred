using LinearAlgebra, Statistics
using JLD2, BSON
using Flux, CUDA
using Plots
using Flux: Data.DataLoader, update!, onehot, onehotbatch, chunk, batchseq
using Zygote
# using Distances:cosine_dist
using Base.Iterators:partition
using RecursiveArrayTools

gr()
theme(:juno)

BSON.@load "saved_models/modelv4_adam_399ep.bson" net
# JLD2.@load "data/moving_forest_v2_8k.jld2" ts as
JLD2.@load "data/moving_forest_8k_training.jld2" train_data

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
    xs = map(x -> reshape(Float32.(x), 256, 20), ts)
    xs, action_labels
end

@inline function get_r(net, x)
    # per batch issue?
    r = prop_err(net, x)
    rhat = ISTA(x, r, net, η=0.01f0, λ=0.001f0, target=0.001f0)
    rhat
end


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
    map(x -> [[x[:,k,:,i] for k in 1:20] for i in 1:f], xy)
end


# xtrain, ytrain = prepare_data("data/moving_forest_v2_8k.jld2", 25)

@load "data/mov_forest_training_data_8k.jld2" xtrain ytrain

##
batchsize = 25
xtest, ytest = prepare_data("data/moving_forest_v2.jld2", batchsize)

##
Rnet = Chain(
    LSTM(100 + 4, 256),
    Dense(256, 100, tanh)
 )

opt = ADAM(0.01)
ps = Flux.params(Rnet)

function loss_fn(x, y)
    sum(Flux.mse.(Rnet.(x), y))
end

Zygote.@nograd function eval_model(xtest, ytest)
    l = loss_fn.(xtest, ytest)
    mean(l)
end

##

function train_model!()
    for (i, (x, y)) in enumerate(zip(xtrain, ytrain))

        g = gradient(() -> loss_fn(x, y), ps)
        update!(opt, ps, g)
        Flux.reset!(Rnet)
    end
end

eval_model(xtest, ytest)

##

out = Rnet.(xtest[2]) .|> x -> net(x)

tmp = [out[k][:,1] for k in 1:20]

preds = reshape.(tmp, 16, 16)

heatmap(preds[9])
x = xs[2]
xx = reshape(x, 16, 16, 20)

begin
    i = 3
    l = @layout [a b]
    im1 = heatmap(xx[:,:,i])
    im2 = heatmap(preds[i] |> cpu)
    plot(im1, im2, layout=l)
end

plot(Ls)

