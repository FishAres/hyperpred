using LinearAlgebra, Statistics
using JLD2, BSON
using Flux, CUDA
using Plots
using Flux: Data.DataLoader, update!
using Zygote
using Distances: cosine_dist
plotlyjs()
theme(:juno)

BSON.@load "saved_models/modelv3_399ep.bson" net
JLD2.@load "data/moving_forest_v2.jld2" ts as

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

##
# 20 step sequences
ind = 4
x = ts[ind] |> Flux.flatten .|> Float32
a = as[ind] .|> Int32

function get_r(net, x)
    r = prop_err(net, x)
    rhat = ISTA(x, r, net, η=0.01f0, λ=0.001f0, target=0.001f0)
    rhat
end

out = reduce(hcat, [get_r(net, x[:,i]) for i in 1:20])
heatmap(out)

##

plot(a)

# quick_anim(permutedims(ts[ind], [3,1,2]), fps=2)

get_r(net, randn(Float32, 256) |> sparsify) |> net |> unwrap |> heatmap
heatmap(net.W)

plot(out[:,1])
plot!(out[:,2])
plot!(out[:,3])
