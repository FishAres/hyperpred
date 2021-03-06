## 
using LinearAlgebra, Statistics
using JLD2
using Flux, CUDA, Distributions
using Plots
using Flux: destructure, Data.DataLoader, update!
using Zygote

CUDA.allowscalar(false)
includet("utils.jl")

# @load "training_forest.jld2" train_data
# @load "testing_forest.jld2" test_data

batchsize = 50
@load "data/training_forest.jld2" train_data

static_data = Flux.unsqueeze(train_data[4,:,:,:], 3)

train_loader = DataLoader(Float32.(static_data), batchsize=batchsize,
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
    net.layers[1].W' * x
end

function train!(net, opt, loader, r)
    scaling = 1 / loader.nobs # 1/N to compute mean loss
    loss_ = 0.0
    strt = time()
    
    for (i, xs) in enumerate(loader)
        x = xs |> Flux.flatten |> normalize |> gpu 
        
        r = Zygote.ignore() do
            r =  @as r̂ r begin
                r̂ -> ISTA(x, r̂, pnet, η=0.01, λ=0.001f0, target=0.5f0)
                norm_rf!
            end
        end
    
        grad = pc_gradient(net, x, r)
        update!(opt, Flux.params(net), grad)
        isnan(l_) && dumb_print("NaN encountered") && break
    end
    loss_ = round(loss_, digits=6)
    tot_time = round(time() - strt, digits=3)
    println("Trained 1 epoch, loss = $loss_, took $tot_time s")

end

## 

Z = 64
pnet = Chain(
    Dense(Z, 256),
    x -> tanh.(x)
) |> gpu

opt = ADAM(0.01)
epochs = 40

for epoch in 1:epochs
    r = Float32.(randn(Z, train_loader.batchsize))  |> gpu
    train!(pnet, opt, train_loader, r)
    println("Finished epoch $epoch")
end

##


prop_err(pnet, x)
##

# @time train(pnet, opt, train_loader, 3)

##
dev = gpu

xs = first(train_loader)
x = xs |> Flux.flatten |> dev
r = Float32.(randn(Z, batchsize)) |> dev

@time rhat = ISTA(x, r, pnet, η=0.01f0, λ=0.001f0, target=0.2f0)
sparsity(rhat)


pred = pnet(rhat)
loss(pred, x)

imshow_Wcol(rand(1:Z), pnet.W |> cpu)

imshow_Wcol(rand(1:batchsize), pred |> cpu)