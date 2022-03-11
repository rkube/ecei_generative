# Utility function

export plot_fake, fake_image, fake_image_3d, conv_layer_size

using Plots
using Statistics
using FileIO
using Images

function plot_fake(G, args, epoch; num_samples=5)
    noise = randn(args["latent_dim"], num_samples) |> gpu;
    fake_x = G(noise) |> cpu;
    for sample in 1:num_samples
        p = contourf(fake_x[:,:,1,sample]', aspectratio=1)
        fname = @sprintf "epoch_%03d_sample_%02d.png" epoch sample
        savefig(p, fname)
    end

    p = contourf(fake_x[:,:,1,1]', aspectratio=1)
end


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
    img_array = zeros(Gray, 24 * num_samples, 8 * args["num_depth"] );
    for s in 1:num_samples
        for c in 1:args["num_depth"]
            img_array[24 * (s - 1) + 1:24 * s, 8 * (c - 1) + 1 : 8 * c] = colorview(Gray, x_fake[:, :, c, s])
        end
    end
    img_array = map(clamp01nan, img_array);    
end
    

# Calculates the size of a convolution layer given
# W: Width of the input array
# K: Kernel size
# S: Stride
# No padding
conv_layer_size(W, K, S) = floor((W-K)/S) + 1|> Int

# Calculates the size of a transpose convolution layer given
# W: Width of input array
# K: Kernel size
# S: Stride
# No padding
trconv_layer_size(W, K, S) = floor((W-1)*S) + K |> Int




