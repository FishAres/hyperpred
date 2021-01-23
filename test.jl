using LinearAlgebra
using Zygote, Flux
using Flux.Data: DataLoader
using Flux: batch, update!
using CUDA
using Plots
using Parameters
using JLD2
using Tullio
using Distributions

include("utils.jl")

CUDA.allowscalar(true)

@with_kw mutable struct Arg
    E::Int = 100
    batchsize::Int = 100
    K::Int64 = 600
    M::Int64 = 256
    Z::Int64 = 5
    η::Float64 = 0.1
    ηᵥ::Float64 = 0.1
    ηᵣ::Float64 = 0.01
    λ::Float64 = 0.0005
    σ::Float64 = 0.01
    session::String = "m1"
end

# init args
arg = Arg(ηᵣ=0.01, ηᵥ=0.01, η=0.01, session="mixture-4", E=300)

##
# load data
@load "training_forest.jld2" train_data
@load "testing_forest.jld2" test_data
train_loader = DataLoader(train_data, batchsize=arg.batchsize, shuffle=true);

##

##
#meow
net = Net(arg.M, arg.K, arg.Z)

opt = Descent(arg.η)
opt_time = Descent(arg.ηᵥ)

# for (cnt, Img) in enumerate(train_loader)
Img = first(train_loader)
T = size(Img, 1)
K = size(net.U, 2)
batchsize = size(Img, 4)
Img = reshape(Img, (T, :, batchsize))

rₚ = ISTA(Img[1,:,:], zeros(K, batchsize), net.U, η=arg.ηᵣ, λ=arg.λ)
r = ISTA(Img[2,:,:], zeros(K, batchsize), net.U, η=arg.ηᵣ, λ=arg.λ)
W = fit_mixture(r, rₚ, net.V)

rhat = time_prediction(net.V, W, r)

grad = gradient(Flux.params(net.V, net.U)) do 
    l = 0.0
    for t = 3:T
        rhat = time_prediction(net.V, W, r)
        Zygote.ignore() do 
            r = ISTA(Img[t, :, :], copy(rhat), net.U, η=arg.η, λ=arg.λ)            
        end
        l += loss_fn(rhat, r)

        l+= loss_fn(Img[t,:,:], net.U * r)
    end
    l
end

update!(opt, Flux.params(net.U), grad)
update!(opt_time, Flux.params(net.V), grad)

net.U = norm_rf(net.U)

##

using Lazy
function imshow_Wcol(i, W)
    heatmap(
        @as x W[:,i] begin
            reshape(x, 16, 16) 
        end
        )
end

imshow_Wcol(12, net.U)
# 
Z = size(V, 3)
K = size(V, 1)
batchsize = size(r, 2)

# destructure bit       mixing 
M = reshape(V, (:, Z)) * W'
M = reshape(M, (K, K, :))

r1hat = batched_mul(M, reshape(r, (K, 1, :)))[:, 1, :]

size(reshape(r, (K, 1, :)))

size(r)

pred = net.U * r1hat

imshow_Wcol(12, pred)