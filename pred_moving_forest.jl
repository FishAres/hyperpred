using LinearAlgebra, Statistics
using JLD2, BSON
using Flux, CUDA
using Plots
using Flux: Data.DataLoader, update!
using Zygote

theme(:juno)

BSON.@load "saved_models/modelv2_120ep.bson" net
JLD2.@load "data/moving_forest_v1.jld2" ts as

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
ind = 8
x = ts[ind] |> Flux.flatten .|> Float32
a = as[ind] .|> Int32
# heatmap(x)
r = prop_err(net, x[:,1])
histogram(r[:])
rhat = ISTA(x, r, net, η=0.01f0, λ=0.001f0, target=0.001f0)
sparsity(rhat)

pred = net(rhat)

begin
    i = 1
    l = @layout [a b]
    im1 = imshow_Wcol(i, pred |> cpu)
    im2 = imshow_Wcol(i, x |> cpu)
    plot(im1, im2, layout=l)
end

##

unwrap(x) = reshape(x, 16, 16)

r_update(r, a, W) = r .+ W * a

function init_rhat(net, x)
    r = prop_err(net, x[:,1])
    rhat = ISTA(x, r, net, η=0.01f0, λ=0.001f0, target=0.001f0)
    # rhat = repeat(rhat, 1, 20)
    rhat
end
opt = Descent(0.01)

action_labels = map(x -> Float32.(Flux.onehotbatch(x, 1:4)), as) |> gpu

##
ind = 8
get_slice(ind) = ts[ind] |> Flux.flatten .|> Float32

r = prop_err(net, x[:,1])
rhat = ISTA(x, r, net, η=0.01f0, λ=0.001f0, target=0.001f0)

pred0 = net(rhat)

a = as[ind] .|> Int32
act_vec = Flux.onehotbatch(a, 1:4) |> Array{Float32}

##

xs = map(x -> reshape(x, 256, 20), ts)

tmp = collect(partition(xs, 32))
