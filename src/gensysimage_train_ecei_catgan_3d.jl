#  2022-04-15 I used this file to generate a sysimage to be used for 
#
# 
# 
using Flux
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold
using CUDA
using Zygote
using Random
using StatsBase
using LinearAlgebra
using Logging
using Combinatorics
using JSON
using FileIO
using ColorSchemes
using Images
using BSON: @save


using ecei_generative

# Set up logging and pycall

open("config.json", "r") do io
    global args = JSON.parse(io)
end


# num_depth is the number of ecei frames bundled together into a single example
data_1 = load_from_hdf(5.9, 6.1, 35000, 50000, "/home/rkube/gpfs/KSTAR/025260", 25260, "GT");
data_2 = load_from_hdf(5.4, 5.6, 5000, 10000, "/home/rkube/gpfs/KSTAR/025880", 25880, "GR");
data_3 = load_from_hdf(2.6, 2.8, 5000, 9000, "/home/rkube/gpfs/KSTAR/022289", 22289, "GT");

data_1_tr = transform_dataset(data_1, args) |> gpu;
data_2_tr = transform_dataset(data_2, args) |> gpu;
data_3_tr = transform_dataset(data_3, args) |> gpu;

# Label the various classes
labels_1 = onehotbatch(repeat([:a], size(data_1_tr)[end]), [:a, :b, :c]) |> gpu;
labels_2 = onehotbatch(repeat([:b], size(data_1_tr)[end]), [:a, :b, :c]) |> gpu;
labels_3 = onehotbatch(repeat([:c], size(data_3_tr)[end]), [:a, :b, :c]) |> gpu;

# Train / test split
split_ratio = 0.5     
# Calculate number of training / validation points from each individual dataset
num_train_1 = round(size(data_1_tr)[end] * split_ratio) |> Int
num_train_2 = round(size(data_2_tr)[end] * split_ratio) |> Int
num_train_3 = round(size(data_3_tr)[end] * split_ratio) |> Int

# One train loader that yields random samples from all data sets
data_train = cat(data_1_tr[:,:,:,:,1:num_train_1], data_2_tr[:,:,:,:,1:num_train_2], data_3_tr[:,:,:,:,1:num_train_3], dims=5);
labels_train = cat(labels_1[:, 1:num_train_1], labels_2[:, 1:num_train_2], labels_3[:, 1:num_train_3], dims=2);
loader_train = DataLoader((data_train, labels_train ), batchsize=args["batch_size"], shuffle=true);

data_test= cat(data_1_tr[:,:,:,:,num_train_1 + 1:end], data_2_tr[:,:,:,:,num_train_2 + 1:end], data_3_tr[:,:,:,:,num_train_3 + 1:end], dims=5);
labels_test = cat(labels_1[:, num_train_1 + 1:end], labels_2[:, num_train_2 + 1:end], labels_3[:, num_train_3 + 1:end], dims=2);
loader_test = DataLoader((data_train, labels_train ), batchsize=args["batch_size"], shuffle=true);

# Loader for verification give samples from individual datasets. These samples are not to be included in loader_train
loader_1 = DataLoader((data_1_tr[:, :, :, :, num_train_1 + 1:end], labels_1[:, num_train_1 + 1:end]), batchsize=args["batch_size"], shuffle=true);
loader_2 = DataLoader((data_2_tr[:, :, :, :, num_train_2 + 1:end], labels_2[:, num_train_2 + 1:end]), batchsize=args["batch_size"], shuffle=true);
loader_3 = DataLoader((data_3_tr[:, :, :, :, num_train_3 + 1:end], labels_3[:, num_train_3 + 1:end]), batchsize=args["batch_size"], shuffle=true);

D = get_cat_discriminator_3d(args) |> gpu;
G = get_generator_3d(args) |> gpu;

opt_D = getfield(Flux, Symbol(args["opt_D"]))(0.0)
opt_G = getfield(Flux, Symbol(args["opt_G"]))(0.0)

ps_D = Flux.params(D);
ps_G = Flux.params(G);

epoch_size = length(loader_train);


num_batch = 1
for epoch ∈ 1:1
    @show epoch
    for (x, y) ∈ loader_train
        this_batch = size(x)[end]
        # Train the discriminator
        testmode!(G);
        trainmode!(D);
        loss_D, back_D = Zygote.pullback(ps_D) do
            # Sample noise and generate a batch of fake data
            y_real = D(x)
            z = randn(Float32, args["latent_dim"], this_batch) |> gpu;
            y_fake = D(G(z))
            loss_D = -H_of_p(y_real) + E_of_H_of_p(y_real) - E_of_H_of_p(y_fake) + args["lambda"] * Flux.Losses.binarycrossentropy(y_real, y)
        end
        # Implement literal transcription of Eq.(7). Then do gradient ascent, i.e. minimize
        # -L(x, θ) by seeding the gradients with -1.0 instead of 1.0
        grads_D = back_D(one(loss_D));
        Flux.update!(opt_D, ps_D, grads_D)

        # Train the generator
        testmode!(D);
        trainmode!(G);
        loss_G, back_G = Zygote.pullback(ps_G) do
            z = randn(Float32, args["latent_dim"], this_batch) |> gpu;
            y_fake = D(G(z));
            loss_G = -H_of_p(y_fake) + E_of_H_of_p(y_fake)
        end
        grads_G = back_G(one(loss_G));
        Flux.update!(opt_G, ps_G, grads_G)

       global num_batch += 1;
    end
end

