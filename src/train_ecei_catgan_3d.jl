# Train a categorical classifier GAN: on ECEi data
# https://arxiv.org/abs/1511.06390
# 
using Flux
using Flux.Data: DataLoader
using Flux: onehotbatch
using CUDA
using Zygote
using Random
using StatsBase
using LinearAlgebra
using Logging
using JSON

using PyCall
using Wandb

using ecei_generative

# Set up logging and pycall

open("config.json", "r") do io
    global args = JSON.parse(io)
end


wb_logger = WandbLogger(project="ecei_catgan_3class", entity="rkube", config=args)
np = pyimport("numpy")                    


# num_depth is the number of ecei frames bundled together into a single example
data_1 = load_from_hdf(2.6, 2.7, 35000, 50000, "/home/rkube/gpfs/KSTAR/025259", 25259, "GT");
#data_2 = load_from_hdf(5.9, 6.0, 5000, 20000, "/home/rkube/gpfs/KSTAR/025879", 25879, "GR");
data_3 = load_from_hdf(2.6, 2.7, 5000, 9000, "/home/rkube/gpfs/KSTAR/022289", 22289, "GT");

# Re-order data_1 and data_2 to have multiple channels per example
num_samples = size(data_1)[end] ÷ args["num_depth"];
data_1 = data_1[:, :, 1:num_samples * args["num_depth"]];
data_1 = reshape(data_1, (24, 8, args["num_depth"], num_samples));

#num_samples = size(data_2)[end] ÷ args["num_depth"];
#data_2 = data_2[:, :, 1:num_samples * args["num_depth"]];
#data_2 = reshape(data_2, (24, 8, args["num_depth"], num_samples));

num_samples = size(data_3)[end] ÷ args["num_depth"];
data_3 = data_3[:, :, 1:num_samples * args["num_depth"]];
data_3 = reshape(data_3, (24, 8, args["num_depth"], num_samples));
data_3[isnan.(data_3)] .= 0f0;

data_all = cat(data_1, data_3, dims=4);
data_all = reshape(data_all, (size(data_all)[1], size(data_all)[2], size(data_all)[3], 1, size(data_all)[end]));

# Scale data_filt to [-1.0; 1.0]
data_all = 2.0 * (data_all .- minimum(data_all)) / (maximum(data_all) - minimum(data_all)) .- 1.0 |> gpu;

# Label the various classes
labels_1 = onehotbatch(repeat([:a], size(data_1)[4]), [:a, :b, :c]) |> gpu;
labels_3 = onehotbatch(repeat([:b], size(data_3)[4]), [:a, :b, :c]) |> gpu;
labels_all = cat(labels_1, labels_3, dims=2);


# Train / test split
split_ratio = 0.8
num_samples = size(data_all)[5]
num_train = round(split_ratio * num_samples) |> Int
idx_all = randperm(num_samples);      # Random indices for all samples
idx_train = idx_all[1:num_train];     # Indices for training set
idx_test = idx_all[num_train:end];    # Indices for test set

loader_train = DataLoader((data_all[:, :, :, :, idx_train], labels_all[:, idx_train]), batchsize=args["batch_size"], shuffle=true);
loader_test = DataLoader((data_all[:, :, :, :, idx_test], labels_all[:, idx_test]), batchsize=args["batch_size"], shuffle=true);

D = get_cat_discriminator_3d(args) |> gpu;
G = get_generator_3d(args) |> gpu;

opt_D = getfield(Flux, Symbol(args["opt_D"]))(args["lr_D"], Tuple(args["beta_D"]));
opt_G = getfield(Flux, Symbol(args["opt_G"]))(args["lr_G"], Tuple(args["beta_D"]));

ps_D = Flux.params(D);
ps_G = Flux.params(G);

epoch_size = length(loader_train);

num_batch = 1
for epoch ∈ 1:args["num_epochs"]
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

        if num_batch % 25 == 0
            (x_test, y_test) = first(loader_test)
            testmode!(G);
            testmode!(D);
            y_real = D(x_test);
            z = randn(Float32, args["latent_dim"], this_batch) |> gpu;
            y_fake = D(G(z));

            xentropy = Flux.Losses.binarycrossentropy(y_real, y_test)

            y_real = y_real |> cpu;
            y_fake = y_fake |> cpu;
            grads_D1 = grads_D[ps_D[1]][:] |> cpu;
            grads_D4 = grads_D[ps_D[4]][:] |> cpu;
            grads_G1 = grads_G[ps_G[1]][:] |> cpu;
            grads_G4 = grads_G[ps_G[4]][:] |> cpu;

            img = fake_image_3d(G, args, 16);
            img = convert(Array{Float32}, img);

            Wandb.log(wb_logger, Dict("batch" => num_batch, 
                                      "crossentropy" => xentropy,
                                      "hist_gradD_1" => Wandb.Histogram(grads_D1),
                                      "hist_gradD_4" => Wandb.Histogram(grads_D4),
                                      "hist_gradG_1" => Wandb.Histogram(grads_G1),
                                      "hist_gradG_4" => Wandb.Histogram(grads_D4),
                                      "hist y_real" => Wandb.Histogram(y_real),
                                      "hist y_fake" => Wandb.Histogram(y_fake),
                                      "H_real" => -H_of_p(y_real),
                                      "E_real" => E_of_H_of_p(y_real),
                                      "E_fake" => E_of_H_of_p(y_fake),
                                      "Generator" => Wandb.Image(img)))
        end
        global num_batch += 1;
    end
end

close(wb_logger)
