using Zygote
using Flux
using Plots: plot, @animate, heatmap!, gif
using Lazy:@as

loss_function(x, y) = Flux.mse(x, y)

function soft_threshold(x, λ)
    relu(x - λ) - relu(-x - λ)
end

function norm_rf!(U)
    U .= U ./ (sqrt.(sum(U.^2, dims=1)) .+ 1f-12)
end

converged(r, r₀; σ=0.01f0) = (norm(r .- r₀) / (norm(r₀) + 1f-12)) < σ

sparsity(x) = 1.0 - mean(x .== 0.0)

function sparse_converged(x; target=0.4f0)
    sparsity(x) <= target
end

S(x) = log(1.0 + x^2)
S′(x) = 2x / (1.0 + x^2)

norm_(x) = x ./ sum(x)

function dumb_print(x)
    "For debugging when you need boolean values"
    println(x)
    true
end

function ISTA_grad(I::AbstractArray{Float32}, net::Any, r::AbstractArray{Float32})
    " options:
    - custom adjoint (d = r * err)
    "
    gradient(() -> loss_function(I, net(r)), Flux.params(r))
end

"meow "
Zygote.@nograd function ISTA(I, r, net;
                    η=0.01,
                    λ=0.001f0,
                    target=0.01f0)

    opt = Descent(η)
    maxiter = 100
    # r₀ = r
    for i in 1:maxiter
        r₀ = r
        grad = ISTA_grad(I, net, r)

        update!(opt, Flux.params(r), grad)
        r = soft_threshold.(r, λ)
        converged(r, r₀, σ=target) && break
        sparse_converged(r[:], target=0.3) && break
    end
    return r
end

function imshow_Wcol(i, W)
    m = sqrt(size(W, 1)) |> Int
    heatmap(
       W[:,i] |> x -> reshape(x, m, m),
       c=:grayC,
    )
end

function quick_anim(data; fps=2, savestring="digabagooool.gif")
    plt = plot()
    anim = @animate for t = 1:size(data, 1)
        heatmap!(data[t, :, :], color=:greys, colorbar=:none)
    end
    gif(anim, savestring, fps=fps)
end

function sparsify(x; θ=0.2f0)
    out = x;
    out[abs.(x) .<= θ] .= zero(x[1])
    out
end

function maprange(s, a, b)
    a₁, a₂ = minimum(a), maximum(a)
    b₁, b₂ = minimum(b), maximum(b)
    return b₁ .+ (s .- a₁) * (b₂ - b₁) / (a₂ - a₁)
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