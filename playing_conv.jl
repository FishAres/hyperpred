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

static_data = Float32.(Flux.unsqueeze(train_data[4,:,:,:], 3))
static_data = @.(2f0 * static_data - 1f0) 

train_loader = DataLoader(static_data, batchsize=batchsize,
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

function train!(net, opt, loader, r₀)
    scaling = 1 / loader.nobs # 1/N to compute mean loss
    " 1 layer deeper - use once per epoch
        next: add a function that does everything inside the loading loop"
    loss_ = 0.0
    strt = time()
    # L = Array{Float32}(undef, loader.nobs)
    for (i, xs) in enumerate(loader)
        x = xs |> gpu

        rhat = Zygote.ignore() do
            ISTA(x, r₀, net, η=0.01, λ=0.001f0, target=0.25f0)
        end
        grad = pc_gradient(net, x, rhat)
        update!(opt, Flux.params(net), grad)
        
        l_ = loss(net(rhat), x)
        isnan(l_) && dumb_print("NaN encountered") && break
        loss_ += scaling * l_
        # L[i] = l_
    end
    loss_ = round(loss_, digits=3)
    tot_time = round(time() - strt, digits=3)
    println("Trained 1 epoch, loss = $loss_, took $tot_time s")
    # return L
end

## 

nthroot(x, n) = x^(1/n)

isqrt(x, n) = @as z x begin
    x -> nthroot(x, n)
    round
    Int
end

Z = 36
lsize = 125
m = isqrt(lsize, 3)
# map(x -> x^3, 1:10)'


net = Chain(
    Dense(Z, lsize, relu),
    x -> reshape(x, m, m, m, batchsize),
    x -> upsample_nearest(x, (4,4)),
    # ConvTranspose((6,6), 1=>lsize, gelu, stride=2, pad=0),
    # BatchNorm(1),
    ConvTranspose((5,5), m=>1, tanh, stride=1, pad=2),

) |> dev

# net(r)



opt = RMSProp(0.01)
epochs = 5
# L = []
for epoch in 1:epochs
    r = Float32.(randn(Z, train_loader.batchsize)) |> sparsify |> gpu
    l = train!(net, opt, train_loader, r)
    # push!(L, l)
    println("Finished epoch $epoch")
end

## 

dev = gpu
xs = first(train_loader) |> dev
r = Float32.(randn(Z, batchsize))  |> dev

rhat = ISTA(xs, r, net, η=0.01f0, λ=0.001f0, target=0.25f0)

plot(rhat[:] |> cpu)
sparsity(rhat[:])

pred = net(rhat)
heatmap(pred[:,:,1,rand(1:batchsize)] |> cpu)