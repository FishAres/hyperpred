##
using LinearAlgebra, Statistics
using JLD2, BSON
using Flux, CUDA, Distributions
using Plots
using Flux: destructure, Data.DataLoader, update!
using Zygote
plotlyjs()

CUDA.allowscalar(false)
includet("utils.jl")

# @load "training_forest.jld2" train_data
# @load "testing_forest.jld2" test_data

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

function pc_gradient(net, x, r)
    grad = gradient(Flux.params(net)) do
        loss(net(r), x)
    end
    return grad
end

xs = first(train_loader)
T = size(xs, 1) # no. of time slices
Z = 100 # latent dim
wh = size(xs, 2)

loss(x, y) = Flux.mse(x, y)

function prop_err(net, x)
    net.W' * x
end

function train!(net, opt, loader)
    loss_ = 0.0
    strt = time()
    for (i, xs) in enumerate(loader)
        x = xs |> Flux.flatten |> normalize |> dev
        r̂ = prop_err(net, x)

        r = Zygote.ignore() do
            r =  @as r̂ r̂ begin
                r -> ISTA(x, r̂, net, η=0.001, λ=0.0005f0, target=0.001f0)
                # norm_rf!
            end
        end

        grad = pc_gradient(net, x, r)
        update!(opt, Flux.params(net), grad)
        norm_rf!(net.W)
        l_ = loss(net(r), x)
        isnan(l_) && dumb_print("NaN encountered") && break

    end
    loss_ = round(loss_, digits=6)
    tot_time = round(time() - strt, digits=3)
    println("Trained 1 epoch, loss = $loss_, took $tot_time s")

end

##
dev = gpu
Z = 100
pnet = Dense(Z, 256) |> dev
opt = RMSProp(0.01)
epochs = 100

for epoch in 1:epochs
    train!(pnet, opt, train_loader)
    println("Finished epoch $epoch")
end

##
xs = first(test_loader)
x = xs |> Flux.flatten |> dev
r = (prop_err(pnet, x))
rhat = ISTA(x, r, pnet, η=0.01f0, λ=0.001f0, target=0.001f0)
println("Sparsity : $(sparsity(rhat))")

pred = pnet(rhat)
println("Loss : $(loss(pred, x)) ")
# imshow_Wcol(rand(1:Z), pnet.W |> cpu)
# imshow_Wcol(rand(1:batchsize), pred |> cpu)

begin
    i = 3
    l = @layout [a b]
    im1 = imshow_Wcol(i, pred |> cpu)
    im2 = imshow_Wcol(i, x |> cpu)
    plot(im1, im2, layout=l)
end

##

net = cpu(pnet)

# using BSON: @save, @load
BSON.@save "saved_models/modelv2_100ep.bson" net

# BSON.@load "saved_models/firstmodel.bson" pnet

