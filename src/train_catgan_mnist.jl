# Train CatGAN on MNIST to test implementation


using Flux
using Flux.Data: DataLoader
using Flux: onehotbatch
using CUDA
using Zygote
using StatsBase
using Random
using LinearAlgebra
using Logging
using ColorSchemes
using Images

using MLDatasets

# using ecei_generative


function H_of_p(y)
    # Calculate Η_X[p(y|D)]. 
    # Average over the probability of the different examples in the current mini-batch
    mm = mean(y, dims=2)
    # Calculate the entropy by summing by summing over the different classes that y can assume
    -sum(mm .* log.(mm .+ eps(eltype(y))))
end

# Conditional entropy, Eqs. (4) and (5)
function E_of_H_of_p(y)
    # Calculate expectation value of the entropy. Here we first calculate the entropy *capital Eta* Η
    # for individual samples before averaging.
    H = -y .* log.(y .+ eps(eltype(y)))
    # Now sum over the class dimension and average over the sample dimension
    sum(H) / size(H, 2)
end


# Draw 10x10 images, return colorview
function fake_image_mnist(G, latent_dim; img_width=28, img_height=28, num_rows=10, num_cols=10)
    num_samples = num_rows * num_cols;

    noise = randn(Float32, (latent_dim, 1, 1, num_samples)) |> gpu;
    x_fake = G(noise) |> cpu;
    x_fake = x_fake[:, :, 1, :];
    img_array = zeros(Float32, img_height * num_rows, img_width*num_cols);
    for row ∈ 1:num_rows
        for col ∈ 1:num_cols
            img_array[((row - 1) * img_height + 1):(row * img_height),
                      ((col - 1) * img_width + 1):(col * img_width)] = permutedims(x_fake[:, :, ((row -1)* num_cols) + col], (2, 1))
        end
    end
    colorview(Gray, img_array)
end

"""
    Implementation of 
    Unsupervised and semi-supervised learning with Categorical Generative Adversarial Networks
    Springenberg, 2016
"""


# Load MNIST as a training set
all_img_x, all_labels = MNIST(:train)[:]
all_img_x = Flux.unsqueeze(all_img_x, 3);
all_img_gpu = all_img_x |> gpu;


# Train / test split
split_ratio = 0.8
num_samples = size(all_img_x)[4]
num_train = Int(round(split_ratio * num_samples))
idx_all = randperm(num_samples)      # Random indices for all samples
idx_train = idx_all[1:num_train];
idx_test = idx_all[num_train:end];


# Build discriminator and generator CNNs used in the paper
D = Chain(Conv((5, 5), 1=>32, relu),
          MaxPool((3, 3), stride=2),
          Conv((3, 3), 32=>64, relu),
          Conv((3, 3), 64=>64, relu),
          MaxPool((3, 3), stride=2),
          Conv((3, 3), 64=>128, relu),
          Conv((1, 1), 128 => 10, relu),
          Flux.flatten,
          Dense(10, 128, relu),
          Dense(128, 10),
          x -> softmax(x)) |> gpu;

G = Chain(Dense(128, 6144, relu),
          x -> reshape(x, (8, 8, 96, size(x)[4])),
          #x -> Flux.unsqueeze(x, dims=4),
          ConvTranspose((2, 2), 96 => 96, dilation=8, relu),
          Conv((5, 5), 96 => 64, relu, pad=SamePad()),
          ConvTranspose((2, 2), 64 => 64, dilation=12, relu),
          Conv((5, 5), 64 => 64, relu, pad=SamePad()),
          Conv((5, 5), 64 => 1, relu, pad=SamePad())) |> gpu;

# Training parameters
batch_size = 64
lr = 1e-3
latent_dim = 128
λ = 1e-4
num_epochs = 2

opt_D = ADAM(lr)
ps_D = Flux.params(D);

opt_G = ADAM(lr)
ps_G = Flux.params(G);

loader_train = DataLoader((all_img_gpu[:, :, :, idx_train], all_labels[idx_train]), batchsize=batch_size, shuffle=true);
loader_test = DataLoader((all_img_gpu[:, :, :, idx_test], all_labels[idx_test]), batchsize=batch_size, shuffle=true);

epoch_size = length(loader_train);

lossvec_D = zeros(num_epochs)
lossvec_G = zeros(num_epochs)

for epoch ∈ 1:num_epochs
    num_batch = 1;
    @show epoch
    for (x, y) ∈ loader_train 
        y = CuArray(Flux.onehotbatch(y, 0:9))
        this_batch = size(x)[end]
        #testmode!(G);
        #trainmode!(D);
        z = randn(Float32, (latent_dim, 1, 1, this_batch)) |> gpu;
        x_fake = G(z);
        loss_D, back_D = Zygote.pullback(ps_D) do
            # Sample noise and generate a batch of fake data
            y_real = D(x)
            
            y_fake = D(x_fake)
            loss_D = -H_of_p(y_real) + E_of_H_of_p(y_real) - E_of_H_of_p(y_fake) + λ * Flux.Losses.binarycrossentropy(y_real,  y)
            Zygote.ignore() do
                lossvec_D[epoch] = loss_D
            end
        end
        # Implement literal transcription of Eq.(7). Then do gradient ascent, i.e. minimize
        # -L(x, θ) by seeding the gradients with -1.0 instead of 1.0
        grads_D = back_D(one(loss_D));
        Flux.update!(opt_D, ps_D, grads_D)

        # Train the generator
        #testmode!(D);
        #trainmode!(G);
        loss_G, back_G = Zygote.pullback(ps_G) do
            z = randn(Float32, (latent_dim, 1, 1, this_batch)) |> gpu;
            y_fake = D(G(z));
            loss_G = -H_of_p(y_fake) + E_of_H_of_p(y_fake)

            Zygote.ignore() do
                lossvec_G[epoch] = loss_G
            end
        end
        grads_G = back_G(one(loss_G));
        Flux.update!(opt_G, ps_G, grads_G)

        num_batch += 1
    end
    #     # if num_batch % 50 == 0
    # (x_test, y_test) = first(loader_test)
    # this_batch = size(x_test)[end]
    # y_test = CuArray(Flux.onehotbatch(y_test, 0:9));
    # #testmode!(G);
    # #testmode!(D);
    # y_real = D(x_test);
    # z = randn(Float32,  (latent_dim, 1, 1, this_batch)) |> gpu;
    # y_fake = D(G(z));

    # xentropy = λ * Flux.Losses.binarycrossentropy(y_real, y_test)

    # y_real = y_real |> cpu;
    # y_fake = y_fake |> cpu;

    G_img = fake_image_mnist(G, latent_dim);
    save("catgan_mnist_epoch_$(epoch).png", G_img)

end


