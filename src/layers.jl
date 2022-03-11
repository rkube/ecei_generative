
export MinibatchDiscrimination

# Struct that stores the trainable parameter tenor
struct MinibatchDiscrimination{N<:AbstractArray}
    T::N            # This is a 3-dimensional tensor with dimensions
                    # 1 - input_features
                    # 2 - output_features
                    # 3 - intermediate_features
end

# Constructor that creates a MinibatchDiscrimination object given 3 integers that define it's size
function MinibatchDiscrimination(in_features::Integer, out_features::Integer, intermediate_features::Integer)
    return MinibatchDiscrimination(randn((in_features, out_features, intermediate_features)))
end

Flux.@functor MinibatchDiscrimination


function (a::MinibatchDiscrimination)(x::AbstractArray)
    M = x' * reshape(a.T, (size(a.T)[1], :))
    M = reshape(M, (1, :, size(a.T)[2], size(a.T)[3]))
    M_t = permutedims(M, (3, 4, 2, 1))
    M = permutedims(M, (3, 4, 1, 2))
    MMt = M .- M_t
    # dim3 and 4 are symmetric. Sum over dim3 so that we can call flatten to remove singleton dimensions
    out = sum(exp.(-sum(abs.(MMt), dims=2)), dims=3) .- 1.0
    # Concatenate along the code dimension
    cat(x, flatten(out), dims=1)
end


