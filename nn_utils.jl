using Zygote, Flux
using LinearAlgebra, Statistics

# N = 64
# pconv = Chain(
#     x -> reshape(x, (6, 6, 1, size(x, 4))),
#     ConvTranspose((4, 4), 1 => N, relu, stride=1, pad=0),
#     MaxPool((2, 2)),
#     ConvTranspose((4, 4), N => 1, sigmoid, stride=2, pad=0),
# ) 

# r = randn(36, 1, 1, 32)

# function conv_output_size(in_size, filter_size, padding, stride)
#     1 + (in_size - filter_size + 2padding) / stride
# end

function conv_transpose_output_size(in_size, filter_size, padding, stride)
    (in_size - 1) * stride + filter_size - 2padding
end

in_size = 6
filter_size = 5
padding = 1
stride = 1

ps = [in_size, filter_size, padding, stride]
conv_transpose_output_size(ps...)
conv_transpose_output_size(in_size, filter_size, padding, stride)

gradient((x, y) -> loss_fn(x, y), Flux.params(ps))


loss_fn(x, y) = conv_transpose_output_size(x...) - y

# k = (4, 4)

# conv_output_size.((6, 6), (4, 4), 0, 1)
# conv_transpose_output_size.((6, 6), (4, 4), 0, 1)

# m = ConvTranspose((4, 4), 1 => N, relu, stride=1, pad=0)
# mc = Conv((4, 4), 1 => N, relu, stride=1, pad=0)

