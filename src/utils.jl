# Utility function

export H_of_p, E_of_H_of_p, plot_fake, fake_image

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
    # Generate samples and cut out channel dimension
    noise = randn(Float32, args["latent_dim"], num_samples) |> gpu;
    x_fake = G(noise) |> cpu;
    x_fake = x_fake[:, :, 1, :];
    img_array = zeros(Gray, 24, 8 * num_samples);
    for n in 1:num_samples
        img_array[1:24, 8 * (n - 1) + 1 : 8 * n] = colorview(Gray, permutedims(x_fake[:, :, n], (2,1)))
    end
    img_array = map(clamp01nan, img_array);    
end


#marginalized entropy, Eq. (6)
function H_of_p(y)
    # Calculate Η_X[p(y|D)]. 
    # Average over the probability of the different examples in the current mini-batch
    mm = mean(y, dims=2)
    # Calculate the entropy by summing over the different classes that y can assume
    -sum(mm .* log.(mm .+ eps(eltype(y))))
end

# Conditional entropy, Eqs. (4) and (5)
function E_of_H_of_p(y)
    # Calculate expectation value of the entropy. Here we first calculate the entropy  Η
    # for individual samples before averaging.
    H = -y .* log.(y .+ eps(eltype(y)))
    # Now sum over the class dimension and average over the sample dimension
    sum(H) / size(H, 2)
end

