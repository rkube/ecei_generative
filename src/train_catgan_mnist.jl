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
using Clustering: mutualinfo, ClusteringResult
# using ecei_generative

"""
    Calculates the marginal entropy of the Dataset

    Do this by approximating the marginal entropy as the marginal entropy in the current batch
    
    H_X[p(y|D)] as H[1/B Σ_{x ∈ X} p(y|xⁱ, D)]
"""
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
    # This is just the sum over both dimensions and dividing by the number of samples.
    sum(H) / size(H, 2)
end


# Draw 10x10 images, return colorview
function fake_image_mnist(G, latent_dim; img_width=28, img_height=28, num_rows=10, num_cols=10)
    num_samples = num_rows * num_cols;

    noise = randn(Float32, (1, 1, latent_dim, num_samples)) |> gpu;
    x_fake = G(noise) |> cpu;
    x_fake = x_fake[:, :, 1, :];
    img_array = zeros(Float32, img_height * num_rows, img_width*num_cols);
    for row ∈ 1:num_rows
        for col ∈ 1:num_cols
            img_array[((row - 1) * img_height + 1):(row * img_height),
                      ((col - 1) * img_width + 1):(col * img_width)] = permutedims(x_fake[:, :, ((row -1)* num_cols) + col], (2, 1))
        end
    end
    # Transform range from [-1:1] to 0:1 for color coding
    img_array = (img_array .+ 1f0) .* 5f-1
    colorview(Gray, img_array)
end

struct GroundTruthResult <: ClusteringResult
    assignments::Vector{Int}   # assignments (n)
end

"""
    Implementation of 
    Unsupervised and semi-supervised learning with Categorical Generative Adversarial Networks
    Springenberg, 2016
"""




# Load MNIST as a training set
all_img_x, all_labels = MNIST(:train)[:]
# rescale to -1:1 
all_img_x = 2f0 .* all_img_x .- 1f0;
all_img_x = Flux.unsqueeze(all_img_x, 3);
all_img_gpu = all_img_x |> gpu;

# Train / test split
split_ratio = 0.8
num_samples = size(all_img_x)[4]
num_train = Int(round(split_ratio * num_samples))
idx_all = randperm(num_samples)      # Random indices for all samples
idx_train = idx_all[1:num_train];
idx_test = idx_all[num_train:end];

lrelu(x) = leakyrelu(x, 2f-1);
# Build discriminator and generator CNNs used in the paper
D = Chain(Conv((5, 5), 1=>32, lrelu),
          MaxPool((3, 3), stride=2),
          Conv((3, 3), 32=>64, lrelu),
          Conv((3, 3), 64=>64, lrelu),
          MaxPool((3, 3), stride=2),
          Conv((3, 3), 64=>128, lrelu),
          Conv((1, 1), 128=>128, lrelu),
          Flux.flatten,
          #Dense(128, 128, lrelu),
          Dense(128, 10),
          softmax) |> gpu;

# G = Chain(Dense(128, 6144, lrelu),
#           x -> reshape(x, (8, 8, 96, size(x)[4])),
#           ConvTranspose((2, 2), 96 => 96, dilation=8, lrelu),
#           BatchNorm(96),
#           Conv((5, 5), 96 => 64, lrelu, pad=SamePad()),
#           BatchNorm(64),
#           ConvTranspose((2, 2), 64 => 64, dilation=12, lrelu),
#           BatchNorm(64),
#           Conv((5, 5), 64 => 64, lrelu, pad=SamePad()),
#           Conv((5, 5), 64 => 1, tanh, pad=SamePad())) |> gpu;


# Try getting rid of the dense layer in G

G = Chain(ConvTranspose((5, 5), 128=>128, lrelu),  # 5x5
          BatchNorm(128),
          ConvTranspose((4, 4), 128 => 64, lrelu, stride=2), # 12x12
          BatchNorm(64),
          ConvTranspose((4, 4), 64 => 32, lrelu, stride=2), # 26x26
          BatchNorm(32),
          ConvTranspose((3, 3), 32 => 32, lrelu, stride=1), # 28x28
          BatchNorm(32),
          ConvTranspose((3, 3), 32 =>1, tanh, pad=SamePad())
          ) |> gpu;

# Training parameters
batch_size = 1024
lr = 2e-3
latent_dim = 128
λ = 1e-4
num_epochs = 20

opt_D = RMSProp(lr)
ps_D = Flux.params(D);

opt_G = ADAM(lr, (0.5, 0.999))
ps_G = Flux.params(G);

loader_train = DataLoader((all_img_gpu[:, :, :, idx_train], all_labels[idx_train]), batchsize=batch_size, shuffle=true);
loader_test = DataLoader((all_img_gpu[:, :, :, idx_test], all_labels[idx_test]), batchsize=batch_size, shuffle=true);

epoch_size = length(loader_train);

lossvec_D = zeros(num_epochs)
lossvec_G = zeros(num_epochs)
nmi = zeros(num_epochs)

for epoch ∈ 1:num_epochs
    num_batch = 1;
    @show epoch
    for (x, labels) ∈ loader_train 
        labels = CuArray(Flux.onehotbatch(labels, 0:9));
        this_batch = size(x)[end];
        
        z = randn(Float32, (1, 1, latent_dim, this_batch)) |> gpu;
        x_fake = G(z);
        loss_D, back_D = Zygote.pullback(ps_D) do
            # Generate class assignments for real data
            y_real = D(x)
            # Generate class assignments for fake data
            y_fake = D(x_fake)
            # Note that the sign of the terms are flipped since the paper asks for maximization. We do minimization
            loss_D = -H_of_p(y_real) + E_of_H_of_p(y_real) - E_of_H_of_p(y_fake) - λ * Flux.Losses.binarycrossentropy(labels, y_real)
            Zygote.ignore() do
                lossvec_D[epoch] = loss_D
            end
            return loss_D
        end
        grads_D = back_D(one(loss_D));
        Flux.update!(opt_D, ps_D, grads_D)

        # Train the generator
        #z = randn(Float32, (1, 1, latent_dim, this_batch)) |> gpu;
        loss_G, back_G = Zygote.pullback(ps_G) do
            y_fake = D(G(z));
            loss_G = -H_of_p(y_fake) + E_of_H_of_p(y_fake)

            Zygote.ignore() do
                lossvec_G[epoch] = loss_G
            end
            return loss_G
        end
        grads_G = back_G(one(loss_G));
        Flux.update!(opt_G, ps_G, grads_G)

        num_batch += 1
    end

    # Calculate mutual information
    (x, labels_true) = first(loader_test);
    res_true = GroundTruthResult(labels_true .+ 1);
    labels_D = [ix[1] for ix in argmax(D(x), dims=1)][1,:]
    res_D = GroundTruthResult(labels_D)
    nmi[epoch] = mutualinfo(res_true, res_D)

    @show epoch, lossvec_D[epoch], lossvec_G[epoch], nmi[epoch]
   
    G_img = fake_image_mnist(G, latent_dim);
    G_img = map(clamp01nan, G_img)
    save("catgan_mnist_epoch_$(epoch).png", G_img)
end


