# Train a categorical classifier GAN: on ECEi data
# https://arxiv.org/abs/1511.06390
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
using JSON
using FileIO
using ColorSchemes
using Images
using BSON: @save


using PyCall
using Wandb

using ecei_generative

# Set up logging and pycall

open("config.json", "r") do io
    global args = JSON.parse(io)
end


# num_depth is the number of ecei frames bundled together into a single example
#data_1 = load_from_hdf(2.6, 2.7, 35000, 50000, "/home/rkube/gpfs/KSTAR/025259", 25259, "GT");
data_1 = load_from_hdf(5.9, 6.0, 35000, 50000, "/home/rkube/gpfs/KSTAR/025260", 25260, "GT");
data_2 = load_from_hdf(5.4, 5.5, 5000, 10000, "/home/rkube/gpfs/KSTAR/025880", 25880, "GR");
data_3 = load_from_hdf(2.6, 2.7, 5000, 9000, "/home/rkube/gpfs/KSTAR/022289", 22289, "GT");

data_1_tr = transform_dataset(data_1, args);
data_2_tr = transform_dataset(data_2, args);
data_3_tr = transform_dataset(data_3, args);
data_all = cat(data_1_tr, data_2_tr, data_3_tr, dims=ndims(data_1_tr)) |> gpu;

# Label the various classes
labels_1 = onehotbatch(repeat([:a], size(data_1_tr)[end]), [:a, :b, :c]) |> gpu;
labels_2 = onehotbatch(repeat([:b], size(data_1_tr)[end]), [:a, :b, :c]) |> gpu;
labels_3 = onehotbatch(repeat([:c], size(data_3_tr)[end]), [:a, :b, :c]) |> gpu;
labels_all = cat(labels_1, labels_2, labels_3, dims=2);


# Train / test split
split_ratio = 0.8
num_samples = size(data_all)[end]
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

wb_logger = WandbLogger(project="ecei_catgan_3", entity="rkube", config=args)
np = pyimport("numpy")                    

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
            (x_test, y_test) = first(loader_test);
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

            img = channelview(fake_image_3d(G, args, 16));
            img = permutedims(img, (3,1,2));

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
    D_c = D |> cpu;
    G_c = G |> cpu;
    @save "/home/rkube/gpfs/catgan_epoch$(epoch).bson" D_c G_c
end

close(wb_logger)

# test how well we predict class on the training loader:
(x,y) = first(loader_train);
preds = Flux.onecold(D(x), [:a, :b, :c]);
preds .== Flux.onecold(y, [:a, :b, :c])

D_c = D|> cpu;
G_c = G|> cpu;
@save "/home/rkube/gpfs/catgan.bson" D_c G_c

