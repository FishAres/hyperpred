##
using LinearAlgebra, Statistics
using JLD2, BSON
using Flux, CUDA
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

static_data = train_data # Flux.unsqueeze(train_data[:,:,:,:], 3)
static_test_data = test_data # Flux.unsqueeze(test_data[4,:,:,:], 3)


train_loader = DataLoader(Float32.(static_data), batchsize=batchsize,
                          shuffle=true, partial=false)

test_loader = DataLoader(Float32.(static_test_data), batchsize=batchsize,
                          shuffle=true, partial=false)

##

Z = 5
N = 100
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

function hyper_forward(hnet, pnet, x, Vmix, r)
    w = hnet(r)
    Vw = pred_mix(w, Vmix, Z)

    r̂ = @as x r begin
        batched_mul(Vw, Flux.unsqueeze(x, 2))
        x -> dropdims(x, dims=2)
        x -> Float32.(x)
        x -> relu.(x)
    end

    r′ = ISTA(x, r̂, pnet)
    pred = pnet(r′)

    im_loss = loss_fn(x, pred)
    temp_loss = loss_fn(r′, r̂)
    return im_loss, temp_loss
end

function pred_mix(w, V, Z)
    out = w' * reshape(V, Z, N * N)
    reshape(out, N, N, batchsize)
end

function train_batch(hnet, net, xs, dev=cpu)
    Vmix = 0.0f0 .* randn(Z, N, N) |> dev
    r = Float32.(zeros(N, batchsize)) |> dev
    imloss = 0.0f0
    temploss = 0.0f0
    for t in 1:size(xs, 1)
        x′ = xs[t,:,:,:] |> Flux.flatten |> dev
        l1, l2 = hyper_forward(hnet, net, x′, Vmix, r)
        imloss += l1
        temploss += l2
    end
    grad = gradient(() -> (imloss + temploss), Flux.params(hnet))
    update!(opt, Flux.params(hnet), grad)
    return temploss
end

function train_epoch(hnet, net, loader)
    l = 0.0f0
    pr = 1 / loader.batchsize # for averaging
    for (i, xs) in enumerate(loader)
        l += pr * train_batch(hnet, net, xs, dev)
    end
    return l
end
##

Z = 5
N = 100
dev = gpu

Vmix = 0.01randn(Z, N, N) |> dev
hnet = hypernet(N, Z) |> dev
net = net |> dev
opt = ADAM(0.01)

epochs = 5

for epoch in 1:epochs
    l = train_epoch(hnet, net, train_loader)
    # println("Loss: $l")
end

##

begin
    i = 3
    l = @layout [a b]
    im1 = imshow_Wcol(i, pred |> cpu)
    im2 = imshow_Wcol(i, x |> cpu)
    plot(im1, im2, layout=l)
end

imloss = loss_fn(x, pred)
temp_loss = loss_fn(r′, r̂)

θ, re = destructure(net)

##
i = 4
xs = first(test_loader)
r = Float32.(randn(N, batchsize)) |> dev
x = xs[i,:,:,:] |> Flux.flatten |> dev
rhat = ISTA(x, r, net, η=0.01f0, λ=0.001f0, target=0.001f0)

println("Sparsity : $(sparsity(rhat))")

pred = net(rhat)
println("Loss : $(loss_fn(pred, x)) ")

begin
    i = 3
    l = @layout [a b]
    im1 = imshow_Wcol(i, pred |> cpu)
    im2 = imshow_Wcol(i, x |> cpu)
    plot(im1, im2, layout=l)
end


rf = net.W |> cpu
out_dim, M = 100, 16
rf = reshape(rf, :, out_dim)
# normalize
rf = rf ./ maximum(abs.(rf), dims=1)
rf = reshape(rf, M, M, out_dim)
# plotting
n = Int64(ceil(sqrt(size(rf, 3))))

##

BSON.@load "saved_models/modelv2_100ep.bson" net
