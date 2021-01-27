using Zygote, Flux
using LinearAlgebra, Statistics

# function conv_output_size(in_size, filter_size, padding, stride)
#     1 + (in_size - filter_size + 2padding) / stride
# end

function conv_transpose_output_size(in_size, filter_size, padding, stride)
    (in_size - 1) * stride + filter_size - 2padding
end

