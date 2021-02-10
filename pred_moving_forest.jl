using LinearAlgebra, Statistics
using JLD2, BSON
using Flux, CUDA
using Plots
using Flux: Data.DataLoader, update!, onehot, onehotbatch, chunk, batchseq
using Zygote
using Distances:cosine_dist
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

function get_processed_data(filename)
    @load filename ts as
    action_labels = map(x -> Float32.(Flux.onehotbatch(x, 1:4)), as)
    xs = map(x -> reshape(Float32.(x), 256, 20), ts)
    xs, action_labels
end

xs, as = get_processed_data("data/moving_forest_v2.jld2")

function get_r(net, x)
    r = prop_err(net, x)
    rhat = ISTA(x, r, net, η=0.01f0, λ=0.001f0, target=0.001f0)
    rhat
end

function process_data(x, a)
    r = repeat(get_r(net, x[:,1,:]), 1, 20)
    [r; a]
end

function prepare_data(ins, batchsize)
    xs, as = ins
    N = length(xs)
    Xx = [process_data(xs[i], as[i]) for i in 1:N]
    Xx = convert(Array, VectorOfArray(Xx))
    f = @. Int(floor(N / batchsize))
    Xx = reshape(Xx,  size(Xx)[1:2]..., batchsize, f)
    [[Xx[:,k,:,i] for k in 1:20] for i in 1:f]
end
##
# need better names
@load "data/moving_forest_v2_test.jld2" test_data

xs, as = get_processed_data("data/moving_forest_v2.jld2")

Zygote.@nograd function eval_model(xs, net; device=gpu)
    l = zeros(length(test_data))
    for (x, rvs) in zip(xs, test_data)
        x = x |> device
        r = prop_err(net, x)

        rhat = [ISTA(x[:,i], r[:,i], net, η=0.01f0, λ=0.001f0, target=0.001f0) for i in 1:20]
        l[i] = loss_fn(rvs, rhat)
    end
    l
end

eval_model(xs, pnet)

Rnet = Chain(
    LSTM(100 + 4, 256),
    Dense(256, 100, tanh)
 ) |> gpu
opt = ADAM(0.01)
ps = Flux.params(Rnet)

pnet = net |> gpu

function loss_fn(x, y)
    sum(Flux.mse.(Rnet.(x), y))
end

# Ls = zeros(length(train_data))
for epoch in 1:50
    @time for (i, rvs) in enumerate(train_data)
        x = xs[i] |> gpu
        r = prop_err(pnet, x)
        rhat = [ISTA(x[:,i], r[:,i], pnet, η=0.01f0, λ=0.001f0, target=0.001f0) for i in 1:20] |> gpu
        l = Zygote.ignore() do
            loss_fn(rvs, rhat)
        end
        Ls[i] = l
        g = gradient(() -> loss_fn(rvs, rhat), ps)
        update!(opt, ps, g)
        Flux.reset!(Rnet)
    end
end
##

ind = 2
x = xs[ind]
r = prop_err(net, x)
rhat = [ISTA(x[:,i], r[:,i], net, η=0.01f0, λ=0.001f0, target=0.001f0) for i in 1:20]

Rnet = Rnet |> cpu

out = Rnet.(train_data[ind]) |> x -> net.(x)

ys = [out[k][:,1] for k in 1:20]

preds = reshape.(ys, 16, 16)

heatmap(preds[4])

xx = reshape(x, 16, 16, 20)

begin
    i = 5
    l = @layout [a b]
    im1 = heatmap(xx[:,:,i])
    im2 = heatmap(preds[i])
    plot(im1, im2, layout=l)
end

plot(Ls)