# Train a categorical classifier GAN on ECEi data
# https://arxiv.org/abs/1511.06390
# 
using Flux
using Flux.Data: DataLoader
using Flux: onehotbatch
using CUDA
using Zygote
using StatsBase
using LinearAlgebra
using Logging
using TensorBoardLogger

using ecei_generative

println("Remember to set GKSwstype=nul")

tb_logger = TBLogger("logs/testlog")
                    
args = Dict("batch_size" => 1024, "activation" => "leakyrelu", "activation_alpha" => 0.2, 
            "num_epochs" => 10, "latent_dim" => 64, "lambda" => 1e-1,
            "lr_D" => 0.0002, "lr_G" => 0.0002)

with_logger(tb_logger) do
    @info "hyperparameters" args
end

data_1 = load_from_hdf(2.6, 2.7, 35000, 50000, "/home/rkube/gpfs/KSTAR/025259", 25259, "GT");
data_2 = load_from_hdf(5.9, 6.0, 5000, 20000, "/home/rkube/gpfs/KSTAR/025879", 25879, "GR");
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
labels_1 = onehotbatch(repeat([:a], size(data_1)[1]), [:a, :b]) |> gpu;
labels_2 = onehotbatch(repeat([:b], size(data_1)[1]), [:a, :b]) |> gpu;
labels_all = cat(labels_1, labels_2, dims=2);

train_loader = DataLoader((data_all, labels_all), batchsize=args["batch_size"], shuffle=true);

D = get_cat_discriminator(args) |> gpu;
G = get_dc_generator_v2(args) |> gpu;

opt_D = ADAM(args["lr_D"]);
opt_G = ADAM(args["lr_G"]);

ps_D = Flux.params(D);
ps_G = Flux.params(G);

lossvec_G = zeros(args["num_epochs"]);
lossvec_D = zeros(args["num_epochs"]);

epoch_size = length(train_loader);
H_real = zeros(args["num_epochs"] * length(train_loader));
E_real = zeros(args["num_epochs"] * length(train_loader));
E_fake = zeros(args["num_epochs"] * length(train_loader));

for epoch ∈ 1:args["num_epochs"]
    num_batch = 1;
    @show epoch
    for (x, y) ∈ train_loader
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
            testmode!(G);
            testmode!(D);
            y_real = D(x);
            z = randn(Float32, args["latent_dim"], this_batch) |> gpu;
            y_fake = D(G(z));

            H_real[(epoch - 1) * epoch_size + num_batch] = -H_of_p(y_real)
            E_real[(epoch - 1) * epoch_size + num_batch] = E_of_H_of_p(y_real)
            E_fake[(epoch - 1) * epoch_size + num_batch] = -E_of_H_of_p(y_fake)

            xentropy = args["lambda"] * Flux.Losses.binarycrossentropy(y_real, y)

            y_real = y_real |> cpu;
            y_fake = y_fake |> cpu;

            hist_real = fit(Histogram, y_real[:], 0.0:0.01:1.0);
            hist_real = normalize(hist_real, mode=:density);
            hist_fake = fit(Histogram, y_fake[:], 0.0:0.01:1.0);
            hist_fake = normalize(hist_fake, mode=:density);

            with_logger(tb_logger) do
                @info "histograms" real=(hist_real.edges[1], hist_real.weights) log_step_increment=0
                @info "histograms" fake=(hist_fake.edges[1], hist_fake.weights) log_step_increment=0
                @info "Generator" img=fake_image(G, args, 16) log_step_increment=0
                @info "output" crossentropy=xentropy
                @info "output" H_real=-H_of_p(y_real) E_real=E_of_H_of_p(y_real) E_fake=E_of_H_of_p(y_fake)
            end
        end
        num_batch+=1;
    end
end


#loader_one = DataLoader((data_all[:, :, :, 1:10000], zeros(Float32, 10000)), batchsize=10000, shuffle=false);
#loader_two = DataLoader((data_all[:, :, :, end-10000+1:end], ones(Float32, 10000)), batchsize=10000, shuffle=false);
#(x,y) = first(train_loader)
#y_real = D(x);
#z = randn(Float32, args["latent_dim"], args["batch_size"]) |> gpu;
#y_fake = D(G(z));
#
#windowsize=11
#p = plot(rollmean(E_real, windowsize), label="E_real")
#plot!(p, rollmean(E_fake, windowsize), label="E_fake")
#plot!(p, rollmean(H_real, windowsize), label="H_real")
#Plots.savefig(p, "EH.png")
