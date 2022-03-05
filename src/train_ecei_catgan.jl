# Train a categorical classifier GAN on ECEi data
# https://arxiv.org/abs/1511.06390
# 
using Flux
using Flux.Data: DataLoader
using Flux: onehotbatch
using CUDA
using Zygote
using StatsBase
using Random
using LinearAlgebra
using Logging


using PyCall
using Wandb

using ecei_generative

println("Remember to set GKSwstype=nul")

# Set up logging and pycall
wb_logger = WandbLogger(project="ecei_generative", entity="rkube")
np = pyimport("numpy")                    

args = Dict("batch_size" => 1024, "activation" => "leakyrelu", "activation_alpha" => 0.2, 
            "num_epochs" => 10, "latent_dim" => 64, "lambda" => 1e-0,
            "lr_D" => 0.0005, "lr_G" => 0.0005)

with_logger(wb_logger) do
    @info "hyperparameters" args
end

data_1 = load_from_hdf(2.6, 2.7, 35000, 50000, "/home/rkube/gpfs/KSTAR/025259", 25259, "GT");
#data_2 = load_from_hdf(5.9, 6.0, 5000, 20000, "/home/rkube/gpfs/KSTAR/025879", 25879, "GR");
data_2 = load_from_hdf(4.0, 4.1, 20000, 50000, "/home/rkube/gpfs/KSTAR/024562", 24562, "HT");
data_all = cat(data_1, data_2, dims=1);

# Put batch-dimension last
data_all = permutedims(data_all, (3, 2, 1));
# Rehape into images
data_all = reshape(data_all, (8, 24, 1, :));
# Convert to Float32
data_all = convert.(Float32, data_all);
# Scale data_filt to [-1.0; 1.0]
data_all = 2.0 * (data_all .- minimum(data_all)) / (maximum(data_all) - minimum(data_all)) .- 1.0 |> gpu;

# Label the various classes
labels_1 = onehotbatch(repeat([:a], size(data_1)[3]), [:a, :b]) |> gpu;
labels_2 = onehotbatch(repeat([:b], size(data_1)[3]), [:a, :b]) |> gpu;
labels_all = cat(labels_1, labels_2, dims=2);


# Train / test split
split_ratio = 0.8
num_samples = size(data_all)[4]
num_train = round(split_ratio * num_samples) |> Int
idx_all = randperm(num_samples)      # Random indices for all samples
idx_train = idx_all[1:num_train];
idx_test = idx_all[num_train:end];


loader_train = DataLoader((data_all[:, :, :, idx_train], labels_all[:, idx_train]), batchsize=args["batch_size"], shuffle=true);
loader_test = DataLoader((data_all[:, :, :, idx_test], labels_all[:, idx_test]), batchsize=args["batch_size"], shuffle=true);

D = get_cat_discriminator(args) |> gpu;
G = get_dc_generator_v2(args) |> gpu;

opt_D = ADAM(args["lr_D"]);
opt_G = ADAM(args["lr_G"]);

ps_D = Flux.params(D);
ps_G = Flux.params(G);

epoch_size = length(loader_train);

total_batch = 1
for epoch ∈ 1:args["num_epochs"]
    num_batch = 1;
    @show epoch
    for (x, y) ∈ loader_train 
        this_batch = size(x)[end]
        trainmode!(G);
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
        #testmode!(D);
        #trainmode!(G);
        loss_G, back_G = Zygote.pullback(ps_G) do
            z = randn(Float32, args["latent_dim"], this_batch) |> gpu;
            y_fake = D(G(z));
            loss_G = -H_of_p(y_fake) + E_of_H_of_p(y_fake)
        end
        grads_G = back_G(one(loss_G));
        Flux.update!(opt_G, ps_G, grads_G)

        if num_batch % 50 == 0
            (x_test, y_test) = first(loader_test)
            testmode!(G);
            testmode!(D);
            y_real = D(x_test);
            z = randn(Float32, args["latent_dim"], this_batch) |> gpu;
            y_fake = D(G(z));

            xentropy = args["lambda"] * Flux.Losses.binarycrossentropy(y_real, y_test)

            y_real = y_real |> cpu;
            y_fake = y_fake |> cpu;

            # Use Numpy histograms for wandb
            hist_real = np.histogram(y_real[:], 0.0:0.01:1.0, density=true);
            hist_fake = np.histogram(y_fake[:], 0.0:0.01:1.0, density=true);
            img = fake_image(G, args, 16);
            img = convert(Array{Float32}, img);

            # with_logger(wb_logger) do
            log(wb_logger, Dict("batch" => total_batch, "hist y_real" => Wandb.Histogram(y_real),
                                "hist y_fake" => Wandb.Histogram(y_fake)))
            log(wb_logger, Dict("batch" => total_batch, "crossentropy" => xentropy,
                                "H_real" => -H_of_p(y_real),
                                "E_real" => E_of_H_of_p(y_real),
                                "E_fake" => E_of_H_of_p(y_fake)))
            log(wb_logger, Dict("batch" => total_batch, "Generator" => Wandb.Image(img)))
        end
        num_batch += 1;
        global total_batch += 1;
    end
end


#loader_one = DataLoader((data_all[:, :, :, 1:10000], zeros(Float32, 10000)), batchsize=10000, shuffle=false);
#loader_two = DataLoader((data_all[:, :, :, end-10000+1:end], ones(Float32, 10000)), batchsize=10000, shuffle=false);
#(x,y) = first(loader_train)
#y_real = D(x);
#z = randn(Float32, args["latent_dim"], args["batch_size"]) |> gpu;
#y_fake = D(G(z));
#
#windowsize=11
#p = plot(rollmean(E_real, windowsize), label="E_real")
#plot!(p, rollmean(E_fake, windowsize), label="E_fake")
#plot!(p, rollmean(H_real, windowsize), label="H_real")
#Plots.savefig(p, "EH.png")
