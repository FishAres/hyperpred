pnet = Dense(Z, 400, σ; initW=Flux.kaiming_normal) |> gpu
opt = ADAM(0.01)

function ugh(net, opt, epochs, loader)
    for epoch in 1:epochs
        r = Float32.(randn(Z, loader.batchsize)) |> gpu
        train!(net, opt, loader, r)
        println("Finished epoch $epoch")
    end
end

@profview ugh(pnet, opt, 2, train_loader)

function train!(net, opt, loader, r₀)
    scaling = 1 / loader.nobs # 1/N to compute mean loss
    " 1 layer deeper - use once per epoch
        next: add a function that does everything inside the loading loop"
    loss_ = 0.0
    strt = time()
    for (i, xs) in enumerate(loader)
        x = xs |> Flux.flatten |> gpu

        rhat = Zygote.ignore() do
        ISTA(x, r₀, net, η=0.01, λ=0.001, target=0.5)
        end
        
        grad = pc_gradient(net, x, rhat)
        update!(opt, Flux.params(net), grad)
        
        l_ = loss(net(rhat), x)
    
        if isnan(l_)
            println("NaN encountered")
            break
        end
        loss_ += scaling * l_
    end
    tot_time = round(time() - strt, digits=3)
    println("Trained 1 epoch, loss = $loss_, took $tot_time s")
end


