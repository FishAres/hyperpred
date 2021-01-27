## 
using LinearAlgebra, Statistics
using JLD2, BSON
using Flux, CUDA, Distributions
using Plots
using Flux: destructure, Data.DataLoader, update!
using Zygote
# plotlyjs()

CUDA.allowscalar(false)
includet("utils.jl")

BSON.@load "saved_models/modelv2_100ep.bson" net

batchsize = 50
@load "data/training_forest.jld2" train_data
@load "data/testing_forest.jld2" test_data

static_data = Flux.unsqueeze(train_data[4,:,:,:], 3)
static_test_data = Flux.unsqueeze(test_data[4,:,:,:], 3)


train_loader = DataLoader(Float32.(static_data), batchsize=batchsize,
                          shuffle=true, partial=false)

test_loader = DataLoader(Float32.(static_test_data), batchsize=batchsize,
                          shuffle=true, partial=false)

##

Z = 5
N = 64
sRNN(N, output_size) = Flux.Recur(Flux.RNNCell(N, output_size))

hypernet(N, Z) = Chain(
    sRNN(N, 400),
    Dense(400, 500),
    BatchNorm(500),
    x -> elu.(x),
    Dense(500, 250),
    BatchNorm(250),
    x -> elu.(x),
    Dense(250, 100),
    BatchNorm(100),
    x -> elu.(x),
    Dense(100, Z)
)

Vmix = 0.01randn(Z, N, N)

hnet = hypernet(N, Z)
r = randn(N, batchsize)
w = hnet(r)

function pred_mix(w, V, Z)
    out = w' * reshape(V, Z, N * N)
    reshape(out, N, N, batchsize)
end

Vᵥ = pred_mix(w, Vmix, Z)

r̂ = batched_mul(Vᵥ, Flux.unsqueeze(r, 2))


xs = first(train_loader)
x = xs |> Flux.flatten

permutedims(Vᵥ, [3,1,2])

θ, re = destructure(net)