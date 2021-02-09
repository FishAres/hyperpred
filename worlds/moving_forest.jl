##
using LinearAlgebra, Statistics
using JLD2, NPZ, BSON
# using Plots

CUDA.allowscalar(false)
includet("../utils.jl")

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

orig = npzread("data/forrest.npy")
size(orig)

data = orig[1,:,:,1]
fsize = [16, 16]
bounds = map((x, f) -> Int(floor(x / f)), size(data), fsize)

function get_block(img, pos, fs)
    r = @. Int(floor(fs / 2))
    x, y = pos
    x1, x2 = [x - r[1] + 1, x + r[1]]
    y1, y2 = [y - r[2] + 1, y + r[2]]
    inds = CartesianIndices((x1:x2, y1:y2))
    return img[inds]
end

"XXX Needs fixing "
function pick_move(pos, dir; p=0.0)
    tpos = pos + movements[dir]
    exc = tpos .>= bounds # out of bounds
    if sum(exc) > 1
        if Bool((&)(exc...))
            newdir = rand(["left, down"])
        elseif Bool(exc[1])
            newdir = rand(["left", "up", "down"])
        elseif Bool(exc[2])
            newdir = rand(["up", "left", "right"])
        end
    elseif rand() < p
        newdir = rand(keys(movements))
    else
        newdir = dir
    end
    newdir
end

function make_traj(img, fs; len=20)
    r = @. Int(floor(fs / 2))
    bounds = size(img) .- r
    pos = map((x, y) -> rand(x:y), r, bounds) # random initial position
    dir = rand(keys(movements)) # (index of) random initial heading
    out = zeros(fs..., len) # initialize output array

    # p = 1 / len # probability of changing direction
    p = 0.2 # probability of changing direction
    actions = zeros(len)
    for t in 1:len
        out[:,:,t] = get_block(img, pos, fs)

        dir = pick_move(pos, dir, p=p)
        actions[t] = mov_inds[dir]
        pos += movements[dir]
    end
    return out, actions
end

"XXX Fix"
function get_trajs(data, no_traj, fsize)
    T, A = [], []
    t = 0
    while t < no_traj
        try
            out, actions = make_traj(data, fsize)
            push!(T, out)
            push!(A, actions)
            t += 1
        catch
            nothing
        end
    end
    T, A
    end


ts, as = get_trajs(data, 1000, fsize)
@save "data/moving_forest_v2.jld2" ts as
# quick_anim(permutedims(ts[2], [3,1,2]), fps=1)


# BSON.@load "saved_models/modelv2_100ep.bson" net


