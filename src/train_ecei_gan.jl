# Train a GAN on the ECEI data
#
using Flux
using Flux.Data: DataLoader
using CUDA
using Zygote
using Logging
using TensorBoardLogger
using StatsBase
using LinearAlgebra

using ecei_generative

#ENV["GKSwstype"] = "100"
println("Remember to set GKSwstype=nul")
args = Dict("batch_size" => 64, "activation" => "relu", "activation_alpha" => 0.2, "num_epochs" => 50, "latent_dim" => 32);

tb_logger = TBLogger("tb_logs/run", min_level=Logging.Info)

data_filt = load_from_hdf(2.659, 2.663, 35000, 50000, "/home/rkube/gpfs/KSTAR/025259", 25259, "GT");
# Put batch-dimension last
data_filt = permutedims(data_filt, (3, 2, 1));
# Rehape into images
data_filt = reshape(data_filt, (8, 24, 1, :));
# Convert to Float32
data_filt = convert.(Float32, data_filt);
# Scale data_filt to [-1.0; 1.0]
data_filt = 2.0 * (data_filt .- minimum(data_filt)) / (maximum(data_filt) - minimum(data_filt)) .- 1.0 |> gpu;

train_loader = DataLoader(data_filt, batchsize=args["batch_size"], shuffle=true);

D = get_dc_discriminator(args) |> gpu;
G = get_dc_generator_v2(args) |> gpu;

opt_D = RMSProp(5e-4);
opt_G = RMSProp(5e-4);

ps_D = Flux.params(D);
ps_G = Flux.params(G);

lossvec_G = zeros(args["num_epochs"]);
lossvec_D = zeros(args["num_epochs"]);

for epoch in 1:args["num_epochs"]
    loss_sum_D = 0.0f0
    loss_sum_G = 0.0f0

    batch_idx=1

    for real_data in train_loader
        # Train the discriminator
        this_batch = size(real_data)[end];
        noise = randn(Float32, args["latent_dim"], this_batch) |> gpu;
        fake_data = G(noise);
        # Concatenate real and fake data into a single array
        all_data = cat(real_data, fake_data, dims=4);
        # Create a target vector for the discriminator
        all_targets = [ones(eltype(real_data), 1, this_batch) zeros(eltype(fake_data), 1, this_batch)] |> gpu;

        # Evaluate loss function of the Discriminator and pullback gradient
        trainmode!(D)
        loss, back = Zygote.pullback(ps_D) do
            preds = D(all_data);
            loss = Flux.Losses.binarycrossentropy(preds, all_targets);
        end
        lossvec_D[epoch] += loss;

        # Backprop and update parameters of the discriminator
        grads = back(1f0);
        Flux.update!(opt_D, ps_D, grads);
        # Log discriminator gradients
        # Log gradients of the generator, skip the batchnorm layers
        if batch_idx == 1
            i = 0
            @time for p in ps_D[[1, 4, 7, 10]]
                # Fit and normalize a histogram
                hh = fit(Histogram, grads[p][:], -1.0:0.001:1.0)
                normalize(hh, mode=:density)
                with_logger(tb_logger) do
                    @info "D_layer$(i)" hh log_step_increment=0
                end
                i += 1
            end
        end

        testmode!(D);
        # Train the generator
        noise = randn(Float32, args["latent_dim"], this_batch) |> gpu;
        loss, back = Zygote.pullback(ps_G) do
            preds = D(G(noise));
            loss = Flux.Losses.binarycrossentropy(preds, 1f0);
        end
        lossvec_G[epoch] += loss;
        grads = back(1.0f0);
        Flux.update!(opt_G, ps_G, grads)

        # Log gradients of the generator, skip the batchnorm layers
        if batch_idx == 1
            i = 0
            @time for p in ps_G[[1, 4, 7, 10]]
                hh = fit(Histogram, grads[p][:], -1.0:0.001:1.0)
                normalize(hh, mode=:density)
                with_logger(tb_logger) do
                    @info "G_layer$(i)" hh log_step_increment=0
                end
                i += 1
            end
        end
    batch_idx += 1
    end
    lossvec_D[epoch] /= size(data_filt)[end]
    lossvec_G[epoch] /= size(data_filt)[end]
    @show epoch, lossvec_D[epoch], lossvec_G[epoch]
    with_logger(tb_logger) do
        @info "D_loss" lossvec_D[epoch] log_step_increment=0
        @info "G_loss" lossvec_G[epoch] 
    end

    p = plot_fake(G, args, epoch; num_samples=1)
    #@info "image"
end


