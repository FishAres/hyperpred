using LinearAlgebra, Statistics, Random
using NPZ, JLD2
using Plots

data = npzread("forrest.npy");

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

plt = plot()
anim = @animate for t = 1:tlength
    heatmap!(plt, clips[t,:,:,3460], color=:greys, colorbar=:none)
end
gif(anim, fps=2)

idx = shuffle(Array(1:size(clips, 4)))
clips = clips[:,:,:,idx]

split = Int64(ceil(size(clips, 4) * 0.9))
train_data = clips[:,:,:,1:split]
test_data = clips[:,:,:,split + 1:end]

@save "training_forest.jld2" train_data
@save "testing_forest.jld2" test_data