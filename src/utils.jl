# Utility function

export plot_fake, fake_image, fake_image_3d, map_data_to_labels, conv_layer_size

using Statistics
using FileIO
using ColorSchemes
using Images
using Combinatorics

function fake_image(G, args, num_samples)
    # Generate num_samples samples from the generator G, concatenated along horizontal dimension
    noise = randn(Float32, args["latent_dim"], num_samples) |> gpu;
    x_fake = G(noise) |> cpu;
    x_fake = x_fake[:, :, 1, :];
    img_array = zeros(Gray, 24, 8 * num_samples);
    for n in 1:num_samples
        img_array[1:24, 8 * (n - 1) + 1 : 8 * n] = colorview(Gray, permutedims(x_fake[:, :, n], (2,1)))
    end
    img_array = map(clamp01nan, img_array);    
end


function fake_image_3d(G, args, num_samples)
    # Generate samples sequences from the generator G.
    #
    # Image will be structured like this:
    #
    #    img_{1, t=1}   img_{1, t=2}    ...  img_{1, t=num_depth}
    #    img_{2, t=1}   img_{t, t=2}    ...  img_{2, t=num_depth}
    #    ...
    #    img_{num_samples, t=1} img_{num_samples, t=2} ... img_{num_samples,t=num_depth}
    #
    # where each img is a 24x8 image

    noise = randn(Float32, args["latent_dim"], num_samples) |> gpu;
    x_fake = G(noise) |> cpu;
    x_fake = x_fake[:, :, :, 1, :];
    img_array = zeros(RGB, 24 * num_samples, 8 * args["num_depth"] );
    for s in 1:num_samples
        for c in 1:args["num_depth"]
            img_array[24 * (s - 1) + 1:24 * s, 8 * (c - 1) + 1 : 8 * c] = get(ColorSchemes.berlin, x_fake[:, :, c, s])
        end
    end
    img_array = map(clamp01nan, img_array);
end



# Find the mapping of data to cluster assignment that maximizes the number of correct assignments across classes
function map_data_to_labels(data_list, labels)
    maxsum = -1
    best_idx = -1
    now_idx = 1
    for perm in permutations(labels)
        # Calculate the sum of correct assignments using this permutation and keep score.
        # Permutation with the largest sum wins
        current_sum = sum([sum(y .== p) for (y,p) in zip(data_list, perm)])

        if current_sum > maxsum
            maxsum = current_sum
            best_idx = now_idx
        end
        now_idx += 1
    end
    cluster_perm = collect(permutations(labels))[best_idx]
    return(cluster_perm)
end

# Calculates the size of a convolution layer given
# W: Width of the input array
# K: Kernel size
# P: Padding
# S: Stride
conv_layer_size(W::Int, K::Int, P::Int, S::Int) = floor((W + 2P - K)/S) + 1|> Int
conv_layer_size(W, K, S) = conv_layer_size(W, K, 0, S)

# Calculates the size of a transpose convolution layer given
# W: Width of input array
# K: Kernel size
# P: Padding
# S: Stride
trconv_layer_size(W::Int, K::Int, P::Int, S::Int) = floor((W-1)*S) - 2*P + K |> Int
trconv_layer_size(W, K, S) = trconv_layer_size(W, K, 0, S)




