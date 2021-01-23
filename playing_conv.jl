## 
using LinearAlgebra, Statistics
using JLD2
using Flux, CUDA, Distributions
using Plots
using Flux: destructure, Data.DataLoader, update!
using Zygote

CUDA.allowscalar(false)

include("utils.jl")

# @load "training_forest.jld2" train_data
# @load "testing_forest.jld2" test_data

batchsize = 32
@load "train_mnist.jld2" train_data
train_loader = DataLoader(train_data, batchsize=batchsize,
                          shuffle=true, partial=false)

## 

# function pc_loss(x, r, pred)

function pc_gradient(net, x, pred, r)
    grad = gradient(Flux.params(net)) do
        err = x .- pred
        Zygote.ignore() do
            rhat = ISTA(x, r, net)
        end
        pred = net(rhat)
        l = loss(pred, x)
    end
    return grad, pred
end

##

xs = first(train_loader)
T = size(xs, 1) # no. of time slices
Z = 36 # latent dim
batchsize = size(xs, 4)
wh = size(xs, 2)

N = 64
pconv = Chain(
    x -> reshape(x, (6, 6, 1, size(x, 4))),
    ConvTranspose((4,4), 1 => N, relu, stride=1, pad=0),
    MaxPool((2,2)),
    ConvTranspose((4,4), N => 1, sigmoid, stride=2, pad=0),
) |> gpu

opt = RMSProp(0.01)
loss(x, y) = Flux.mse(x, y)

function train(net, opt, loader, epochs)
    scaling = 1 / loader.nobs # 1/N to compute mean loss
    for epoch in 1:epochs
        r₀ = randn(Z, 1, 1, loader.batchsize)
        r = Float32.(deepcopy(r₀)) |> gpu
        pred₀ = net(r)
        pred = Float32.(pred₀)
        loss_ = 0.0
        for (i, xs) in enumerate(loader)
            x = Float32.(Flux.unsqueeze(xs[4,:,:,:], 3)) |> gpu
            grad, pred = pc_gradient(net, x, pred, r)
            update!(opt, Flux.params(net), grad)
            loss_ += scaling * loss(pred, x)
        end
        println("Trained $epoch epochs, loss = $loss_")
    end
end

##

train(pconv, opt, train_loader, 20)

##
xs = first(train_loader)
x = Float32.(Flux.unsqueeze(xs[10,:,:,:], 3)) |> gpu

r = Float32.(randn(Z, 1, 1, batchsize)) |> gpu

# gradient(() -> loss(x, pconv(rhat)), Flux.params(r))

rhat = Float32.(ISTA(x, r, pconv, optimizer=Descent))

pred = pconv(rhat)

heatmap(pred[:,:,1, rand(1:batchsize)] |> cpu)