using LinearAlgebra, Statistics, Random
using NPZ, JLD2

function load_data()
    data = npzread("data/forrest.npy")
    return data
end

function forest_7timeslice(data; save=true)
    "Preston's function"
    tlength = 7
    interval = 5
    img_size = 16
    # time indices
    tidx = collect(1:(tlength * interval):size(data, 1) - (tlength * interval))
    # height indices
    hidx = [img_size * 2:img_size ÷ 2:size(data, 2) - img_size * 2;]
    # width indices
    widx = [img_size * 2:img_size ÷ 2:size(data, 3) - img_size * 2;];
    clips = zeros(tlength, img_size, img_size, length(tidx) * length(hidx) * length(widx))

    counter = 1
    for t in tidx
        tᵢ = [t + idx * interval for idx = 0:tlength - 1]
        for h in hidx
            for w in widx
                clips[:,:,:,counter] = data[tᵢ, h:h + img_size - 1, w:w + img_size - 1]
                counter += 1
            end
        end
    end

    idx = shuffle(Array(1:size(clips, 4)))
    clips = clips[:,:,:,idx]

    split = Int64(ceil(size(clips, 4) * 0.9))
    train_data = clips[:,:,:,1:split]
    test_data = clips[:,:,:,split + 1:end]

    if save
        @save "data/training_forest.jld2" train_data
        @save "data/testing_forest.jld2" test_data
    end
end
"Divide video into [fsize[1], fsize[2], :] frames"
function forest_div(data, fsize; train_fr=0.75, save=true)
    out = [get_blocks(data[i,:,:], fsize) for i in 1:size(data, 1)]

    split = Int(floor(train_fr * length(out)))
    train = out[1:split]
    test = out[split + 1:end]
    if save
        @save "data/training_winforest.jld2" train
        @save "data/testing_winforest.jld2" test
    end
end

" Divide MxN image into blocks of fsize[1] x fsize[2]"
function get_blocks(x, fsize)

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


forest_div(dropdims(load_data(), dims=4), [16, 16])