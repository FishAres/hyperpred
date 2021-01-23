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

batchsize = 32
@load "data/train_mnist.jld2" train_data

static_data = Flux.unsqueeze(train_data[4,:,:,:], 3)

train_loader = DataLoader(Float32.(static_data), batchsize=batchsize,
                          shuffle=true, partial=false)

## 

function pc_gradient(net, x, r)
    grad = gradient(Flux.params(net)) do
        l = loss(net(r), x)
    end
    return grad
end

xs = first(train_loader)
T = size(xs, 1) # no. of time slices
Z = 100 # latent dim
wh = size(xs, 2)

pnet = Dense(Z, 400, σ) |> gpu

r = Float32.(randn(Z, batchsize)) |> gpu

opt = ADAM(0.01)
loss(x, y) = Flux.mse(x, y)

function train(net, opt, loader, epochs)
    scaling = 1 / loader.nobs # 1/N to compute mean loss
    for epoch in 1:epochs
        r = Float32.(randn(Z, loader.batchsize)) |> gpu
        pred = Float32.(net(r))
        loss_ = 0.0
  
        strt = time()
        for (i, xs) in enumerate(loader)
            x = Flux.flatten(xs[4, :,:,:]) |> gpu
            
            rhat = Zygote.ignore() do 
                ISTA(x, r, pnet, η=0.01, λ=0.001, target=0.5)
                # ISTA(x, r, net)
            end
            
            grad = pc_gradient(net, x, rhat)
            update!(opt, Flux.params(net), grad)
            l_ = loss(pnet(rhat), x)
            if isnan(l_)
                println("NaN encountered")
                break
            end
            loss_ += scaling * l_
        end
        tot_time = round(time() - strt, digits=3)
        println("Trained $epoch epochs, loss = $loss_, took $tot_time s")
    end
end

##

@time train(pnet, opt, train_loader, 3)

##
xs = first(train_loader)
x = xs |> Flux.flatten |> gpu
r = Float32.(randn(Z, batchsize)) |> gpu

# Choice of ISTA parameters is important
rhat = ISTA(x, r, pnet, η=0.01, λ=0.001, target=0.5)
plot(rhat[:] |> cpu)

pred = pnet(rhat)
imshow_Wcol(rand(1:batchsize), pred |> cpu)