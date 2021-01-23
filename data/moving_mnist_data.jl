using NPZ, JLD2
using LinearAlgebra


train_mnist = npzread("data/train_mnist.npy")
test_mnist = npzread("data/test_mnist.npy")

train_data = permutedims(train_mnist, [2:4 ; 1])
test_data = permutedims(test_mnist, [2:4 ; 1])

@save "data/train_mnist.jld2" train_data
@save "data/test_mnist.jld2" test_data