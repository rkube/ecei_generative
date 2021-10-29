using ecei_generative: get_framedata, train_dscr!, train_gen!
using Flux
using Flux.Data: DataLoader
using Zygote
using Printf
using CUDA


n_features = 24*8
latent_dim = 32
batch_size = 8
output_period = 100
num_epochs = 1000

data = get_framedata() |> gpu;
train_loader = DataLoader(data, batchsize=batch_size, shuffle=true)

opt_dscr = ADAM(2e-4)
opt_gen = ADAM(2e-4)

discriminator = Chain(Dense(n_features, 256, x -> leakyrelu(x, 0.2f0)),
                      Dropout(0.3),
                      Dense(256, 128, x -> leakyrelu(x, 0.2f0)),
                      Dropout(0.3),
                      Dense(128, 1, sigmoid)) |> gpu;

generator = Chain(Dense(latent_dim, 128, x -> leakyrelu(x, 0.2f0)),
                  Dense(128, 128, x -> leakyrelu(x, 0.2f0)),
                  Dense(128, 128, x -> leakyrelu(x, 0.2f0)),
                  Dense(128, n_features, tanh)) |> gpu;

ps_g = Flux.params(generator);
ps_d = Flux.params(discriminator);

# Loss vectors
lossvec_g = zeros(num_epochs);
lossvec_d = zeros(num_epochs);

# Main training loop
for n ∈ 1:num_epochs
    Σ_loss_g = 0.0f0
    Σ_loss_d = 0.0f0

    for x in train_loader
        this_batch = size(x)[end]
        real_data = flatten(x)

        # Generate noise
        noise = randn(latent_dim, this_batch) |> gpu
        fake_data = generator(noise)
        Σ_loss_d += train_dscr!(ps_d, discriminator, real_data, fake_data, this_batch, opt_dscr)
        Σ_loss_g = train_gen!(ps_g, discriminator, generator, opt_gen, latent_dim, batch_size)
    end

    lossvec_d[n] = Σ_loss_d / size(data)[end]
    lossvec_g[n] = Σ_loss_g / size(data)[end]

    if n % output_period == 0
        @show n
    #    noise = randn(latent_dim, 4) |> gpu;
    #    fake_data = reshape(generator(noise), 24, 8*4);
    #    #p = heatmap(fake_data, colormap=:inferno)
    #    print(p)
    end 
end


