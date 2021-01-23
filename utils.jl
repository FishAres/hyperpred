using Zygote
using Flux
using Lazy: @as

loss_fn(x, y) = Flux.mse(x, y)

function soft_threshold(x, λ)
    Float32.(relu(x - λ) - relu(-x - λ))
end

function cuda_relu(x)
    CUDA.max(0.0, x)
end

converged(r, r₀; σ=0.01) = norm(r - r₀) / (norm(r₀) + 1e-12) < σ

sparsity(x) = 1.0 - mean(x .== 0)

function sparse_converged(x; target=0.4)
    sparsity(x) <= target
end

S(x) = log(1 + x^2)
S′(x) = 2x / (1 + x^2)

norm_(x) = x ./ sum(x)

function dumb_print(x)
    "For debugging when you need boolean values"
    println(x)
    true
end

function ISTA_grad(I::CuArray{Float32,2}, net::Any, r::CuArray{Float32,2})
    gradient(() -> loss_fn(I, net(r)), Flux.params(r))
end

Zygote.@nograd function ISTA(I, r, net; η=0.01, λ=0.001, target=0.2)
    " Takes way too much time"
    # opt = Descent(η)
    maxiter = 200
    ps = Flux.params(r)
    for i in 1:maxiter
        grad = ISTA_grad(I, net, r)
        r .-= η * grad[r]
        # update!(opt, ps, grad)
        r = norm_(soft_threshold.(r, λ))
        sparse_converged(r[:], target=target) && break
    end
    return r
end

function imshow_Wcol(i, W)
    m = sqrt(size(W, 1)) |> Int
    heatmap(
        @as x W[:,i] begin
        reshape(x, m, m) 
    end)
end

function quick_anim(data)
    plt = plot()
    anim = @animate for t = 1:size(data, 1)
        heatmap!(data[t, :, :], color=:greys, colorbar=:none)
    end
    gif(anim, fps=2)
end
