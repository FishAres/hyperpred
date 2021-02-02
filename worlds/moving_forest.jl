##
using LinearAlgebra, Statistics
using RecursiveArrayTools
using JLD2, BSON
using Flux, CUDA
using Plots
using Flux: Data.DataLoader, update!
using Zygote

CUDA.allowscalar(false)
includet("../utils.jl")

@load "data/training_winforest.jld2" train

train_data = convert(Array, VectorOfArray(train))

# BSON.@load "saved_models/modelv2_100ep.bson" net

ind = 6; heatmap(train_data[:,:,ind,1])

using NPZ
orig = npzread("..data/forrest.npy")

heatmap(orig[1, :,:, 1])

function get_blocks(x, fsize)
    " Divide MxN image into blocks of fsize[1] x fsize[2]"
    # max number of windows
    bounds = map((x, f) -> Int(floor(x / f)), size(x), fsize)
    # initialize
    out = zeros(fsize..., prod(bounds))
    function blockinds(ind, fs)
        "return min, max indices for each block"
        (ind - 1) * fs + 1, ind * fs
    end
    cnt = 1
    ind_array = zeros(Int64, prod(bounds), 2)
    for yind in 1:bounds[2]
        for xind in 1:bounds[1]
            x1, x2 = blockinds(xind, fsize[1])
            y1, y2 = blockinds(yind, fsize[2])

            i = CartesianIndices((x1:x2, y1:y2))
            out[:,:, cnt] .= x[i]
            ind_array[cnt, :] = [xind, yind]
            cnt += 1
        end
    end
    return out, ind_array
end

fsize = (16, 16)

a = get_blocks(orig[1,:,:,1], fsize)
a[2]

function forest_div(data, fsize; train_fr=0.75, save=true)
    "Divide video into [fsize[1], fsize[2], :]
    frames"
    Img, inds = [], []
    for t in 1:size(data, 1)
        out, ind = get_blocks(data[i,:,:], fsize)
        push!(Img, out)
        push!(inds, ind)
    end

    return out
end

b = forest_div(orig[:,:,:,1], fsize)

[a[1] for a in b]