using Parameters, Tullio, ProgressMeter

@with_kw struct Arg
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

arg = Arg(ηᵣ=0.01, ηᵥ=0.01, η=0.1, session="mixture-4", E=300)

loss2(x, x̂) = sum((x .- x̂) .^ 2)


@load "data/training_forest.jld2" train_data
train_loader = DataLoader(train_data, batchsize=arg.batchsize, shuffle=true)

mydev = gpu

Zygote.@nograd function soft_threshold(x, λ)
    λ = fill(λ, size(x)) |> mydev
    return relu.(x - λ) - relu.(-1 * x - λ)
end

Zygote.@nograd converged(r, r₀; σ=0.01) = norm(r - r₀) / (norm(r₀) + 1e-12) < 0.01

Zygote.@nograd function ISTA(I, r, U; η=0.01, λ=0.001) 
    #opt = Descent(η)
    stop = false
    c = 0
    while !stop
        r₀  = copy(r)
        grad = gradient(() -> loss_fn(I, U * r), Flux.params(r))
        r = r - η .* grad[r]
        # spasify
        r = soft_threshold(r, λ) |> mydev
        # check convergence
        stop = converged(r, r₀) 
        c += 1
        if c > 100
            println(norm(r .- r₀) / (norm(r₀) + 1e-12))
        end
    end
    return r    
end

mutable struct Net 
    U
    V
    Net(img_size, num_neuron, Z) = new(
        randn(img_size, num_neuron) * 0.05,
        randn(num_neuron, num_neuron, Z) * 0.025
    )
end

Zygote.@nograd norm_rf(U) = U ./ sqrt.(sum(U.^ 2, dims=1))

net = Net(arg.M, arg.K, arg.Z)
net.U = net.U |> mydev
net.V = net.V |> mydev

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


##
# train
epoch_progress = Progress(arg.E, "Epoch")
opt = Descent(arg.η) |> mydev
opt_time = Descent(arg.ηᵥ) |> mydev
R = nothing
r = nothing
r̂ = nothing
@time for e = 1:10
    train_loss_img = 0.0
    train_loss_time = 0.0
    # p = Progress(length(train_loader), "Training")
    c = 1
    for I in train_loader
        T = size(I, 1)
        batchsize = size(I, 4)
        K = size(net.U, 2)
        I = reshape(I, (T, :, batchsize)) |> mydev
        # fit mixture weights
        rₚ = ISTA(I[1,:,:], zeros(K, batchsize) |> mydev, net.U, η=arg.ηᵣ, λ=arg.λ)
        r = ISTA(I[2,:,:], zeros(K, batchsize) |> mydev, net.U, η=arg.ηᵣ, λ=arg.λ)
        W = fit_mixture(r |> cpu, rₚ |> cpu, net.V |> cpu) |> mydev
        # take gradient
        grad = gradient(Flux.params(net.V, net.U)) do 
            l = 0.0 
            for t = 3:T
                r̂ = time_prediction(net.V, W, r)
                @show r̂
                Zygote.ignore() do 
                    r = ISTA(I[t, :, :], copy(r̂), net.U, η=arg.η, λ=arg.λ)
                end
                l += loss_fn(r̂, r)
                # image loss
                l += loss_fn(I[t, :, :], net.U * r)
                break
            end
            l
        end
        # update weights
        update!(opt, Flux.params(net.U), grad)
        update!(opt_time, Flux.params(net.V), grad)
        # normalize
        net.U = norm_rf(net.U) 
        # if mod(c, 25) == 0
        #     step = (e-1) * length(train_loader) + c
        #     log_value(tblogger_train, "Time Loss", train_loss_time / step, step=step)
        #     log_value(tblogger_train, "Img Loss", train_loss_img / step, step=step)
        # end
        c += 1
        # ProgressMeter.next!(p)
        break
    end
    # RF = net.U |> cpu
    # fig = plot_rf(RF[:, 1:100], 100, 16)
    # log_image(tblogger_train, "rf", fig, step=e * length(train_loader))
    # log_value(tblogger_train, "Time Loss", train_loss_time, step=e * length(train_loader))
    # log_value(tblogger_train, "Img Loss", train_loss_img, step=e * length(train_loader))
    # if mod(e, 2) == 0
    #     RF = net.U |> cpu
    #     fig = plot_rf(RF[:, 1:100], 100, 16)
    #     log_image(tblogger_train, "rf", fig, step=e * length(train_loader))
    #     @save "../RFs/net_$e.jld" net
    # end
    # ProgressMeter.next!(epoch_progress)
    break
end

