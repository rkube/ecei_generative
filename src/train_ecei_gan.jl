# Train a Wasserstein GAN on the ECEI data
#
using Flux
using Flux.Data: DataLoader
using CUDA
using Zygote

using ecei_generative

args = Dict("batch_size" => 32, "num_epochs" => 20, "latent_dim" => 32);

data_filt = load_from_hdf(2.659, 2.660, 35000, 50000, "/home/rkube/gpfs/KSTAR/025259", 25259, "GT");
data_filt = permutedims(data_filt, (3, 2, 1));
data_filt = reshape(data_filt, (8, 24, 1, :));
# TODO: Scale data_filt to [-1.0; 1.0]
# TODO: Convert data_filt to Float32

train_loader = DataLoader(data_filt, batchsize=args["batch_size"], shuffle=true);

D = get_dc_discriminator(args);
G = get_dc_generator(args);

opt_D = ADAM(2e-4);
opt_G = ADAM(2e-4);

ps_D = Flux.params(D);
ps_G = Flux.params(G);

lossvec_G = zeros(args["num_epochs"]);
lossvec_D = zeros(args["num_epochs"]);

for epoch in 1:args["num_epochs"]
    loss_sum_D = 0.0f0
    loss_sum_G = 0.0f0

    for real_data in train_loader
        # Train the discriminator
        this_batch = size(real_data)[end];
        noise = randn(Float32, args["latent_dim"], this_batch);
        fake_data = G(noise);
        # Concatenate real and fake data into a single array
        all_data = cat(real_data, fake_data, dims=4);
        # Create a target vector for the discriminator
        all_targets = [ones(eltype(real_data), 1, this_batch) zeros(eltype(fake_data), 1, this_batch)];

        # Evaluate loss function
        loss, back = Zygote.pullback(ps_D) do
            preds = D(all_data);
            loss = Flux.Losses.binarycrossentropy(preds, all_targets);
        end
        loss_sum_D += loss / this_batch;

        # Backprop and update parameters of the discriminator
        grads = back(1f0)
        Flux.update!(opt_D, ps_D, grads)


        # Train the generator
        noise = randn(args["latent_dim"], this_batch);
        loss, back = Zygote.pullback(ps_G) do
            preds = D(G(noise))
            loss = Flux.Losses.binarycrossentropy(preds, 1f0)
        end
        loss_sum_G += loss / this_batch;
        grads = back(1.0f0);
        Flux.update!(opt_G, ps_G, grads)
    end

end



