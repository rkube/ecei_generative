using ecei_generative: get_framedata, train_decsr!, train_gen!
using Flux
using Flux.Data: DataLoader
using Zygote
using Printf


n_features = 24*8
latent_dim = 64

data = get_framedata()
train_loader = DataLoader(data, batchsize=64, shuffle-true)

opt_dscr = ADAM(2e-4)
opt_gen = ADAM(2e-4)


discriminator = Chain(Dense(n_features, 1024, x -> leakyrelu(x, 0.2f0)),
                      Dropout(0.3),
                      Dense(1024, 512, x -> leakyrelu(x, 0.2f0)),
                      Dropout(0.3),
                      Dense(512, 256, x -> leakyrelu(x, 0.2f0)),
                      Dropout(0.3),
                      Dense(256, 1, sigmoid)) |> gpu

generator = Chain(Dense(latent_dim, 256, x -> leakyrelu(x, 0.2f0)),
                  Dense(256, 512, x -> leakyrelu(x, 0.2f0)),
                  Dense(512, 1024, x -> leakyrelu(x, 0.2f0)),
                  Dense(1024, n_features, tanh)) |> gpu



# Flatten out training loop
loss_sum_gen = 0.0f0
loss_sum_dscr = 0.0f0
x = first(train_loader)
this_batch = size(x)[end]
real_data = flatten(x)

# Generate noise
noise = randn(latent_dim, this_batch) |> gpu
fake_data = generator(noise)
loss_dscr = train_dscr!(discriminator, real_data, fake_data, this_batch)
loss_sum_dscr += loss_dscr

loss_gen = train_gen!(discriminator, generator)
loss_sum_gen += loss_gen

