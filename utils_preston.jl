using Zygote, Tullio
using Flux: mse


loss_fn(x, y) = mse(x, y)

Zygote.@nograd function soft_threshold(x, λ)
    relu.(x .- λ) - relu.(-x .- λ)
end

Zygote.@nograd converged(r, r₀; σ=0.01) = norm(r - r₀) / (norm(r₀) + 1e-12) < 0.01

Zygote.@nograd function ISTA(I, r, net; η=0.01, λ=0.001) 
    # opt = Descent(η)
    stop = false
    c = 0
    while !stop
        r₀  = copy(r)
        grad = gradient(() -> loss_fn(I, net(r)), Flux.params(r))
        update!(opt, Flux.params(r), grad)
        # spasify
        r = soft_threshold(r, λ)
        # check convergence
        stop = converged(r, r₀) 
        c += 1
        if c > 100
            break
            # println(norm(r .- r₀) / (norm(r₀) + 1e-12))
        end
    end
    return r   
end

# network
mutable struct Net 
    U
    V
    Net(img_size, num_neuron, Z) = new(
        randn(img_size, num_neuron) * 0.05,
        randn(num_neuron, num_neuron, Z) * 0.025
    )
end

function plot_rf(rf, out_dim, M)
    rf = reshape(rf, :, out_dim)
    # normalize
    rf = rf ./ maximum(abs.(rf), dims=1)
    rf = reshape(rf, M, M, out_dim)
    # plotting
    n = Int64(ceil(sqrt(size(rf, 3))))
    fig, axes = plt.subplots(nrows=n, ncols=n, sharex=true, sharey=true)
    fig.set_size_inches(10, 10)
    for i in 1:size(rf, 3)
        ax = axes[i]
        ax.imshow(rf[:,:,i], cmap="gray", vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    end
    for j in size(rf, 3) + 1:n * n
        ax = axes[j]
        ax.imshow(ones_like(rf[:,:,1]) .* -1, cmap="gray", vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    end
    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    return fig 
end

Zygote.@nograd norm_rf(U) = U ./ sqrt.(sum(U.^2, dims=1))

function fit_mixture(r, rₚ, V) 
    
    @tullio μ[k,b,z] := V[k,f,z] * rₚ[f,b] 
    μ = relu.(μ)

    Σ = fill(0.01, size(μ))
    W = zeros(size(rₚ, 2), size(V, 3)) 
    for z = 1:size(V, 3)
        for b = 1:size(rₚ, 2)
            @inbounds W[b,z] = logpdf(MvNormal(μ[:,b,z], Σ[:,b,z] |> cpu), r[:, b])
        end
    end
    # normalize
    return W ./ sum(W, dims=2)
end

function time_prediction(V, W, r)
    Z = size(V, 3)
    K = size(V, 1)
    batchsize = size(r, 2)
    M = reshape(V, (:, Z)) * W';
    M = reshape(M, (K, K, :))
    r̂ₜ = batched_mul(M, reshape(r, (K, 1, :)))[:, 1, :];
end

using Lazy
function imshow_Wcol(i, W)
    m = sqrt(size(W, 1)) |> Int
    heatmap(
        @as x W[:,i] begin
        reshape(x, m, m) 
    end
        )
end

function quick_anim(data)
    plt = plot()
    anim = @animate for t = 1:size(data, 1)
        heatmap!(data[t, :, :], color=:greys, colorbar=:none)
    end
    gif(anim, fps=2)
end
