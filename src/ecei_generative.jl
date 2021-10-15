module ecei_generative

using Printf

"""
    Loads frames from hdf5 file
"""
function get_framedata(shotnr=25259, dev="GT", chunk=137)
   # Construct filename and dataset name
   # Files are written on cori, see ~/delta_misc/
   fname = @sprintf "frames_%05d_%s_%03d.h5" shotnr dev chunk
   dset_name = @sprintf "frame_%03d" chunk
   # Open hdf5 file and read dataset
   fid = h5open(fname)
   # Permute dimensions: dim1: rows, dim2: columns, dim3: time
   frame_data = permutedims(fid[dset_name][:,:,:], [3, 2, 1])
   close(fid)

   return frame_data
end


"""
    Train GAN discriminator. This is taken from my mnist_gan package
"""
function train_dscr!(discriminator, real_data, fake_data, this_batch)
    # Given real and fake data, update the parameters of the discriminator network in-place
    # Assume that size(real_data) = 784xthis_batch
    # this_batch is the number of samples in the current batch
    # Concatenate real and fake data into one big vector
    all_data = hcat(real_data, fake_data)
    # Target vector for predictions
    all_target = [ones(eltype(real_data), 1, this_batch) zeros(eltype(fake_data), 1, this_batch)] |> gpu;

    ps = Flux.params(discriminator)
    loss, back = Zygote.pullback(ps) do
        preds = discriminator(all_data)
        # The documentation says to use logitbinarycrossentropy, but for this case the plain
        # binarycrossentropy works fine
        loss = Flux.Losses.binarycrossentropy(preds, all_target)
    end
    # To get the gradients we evaluate the pullback with 1.0 as a seed gradient.
    grads = back(1f0)

    # Update the parameters of the discriminator with the gradients we calculated above
    Flux.update!(opt_dscr, Flux.params(discriminator), grads)

    return loss
end


"""
    Train GAN generator. This is taken from my mnist_gan package
"""
function train_gen!(discriminator, generator)
    # Updates the parameters of the generator in-place
    # Let the generator create fake data which should out-smart the discriminator
    # The discriminator is fooled if it outputs a 1 for the samples generated
    # by the generator.
    noise = randn(latent_dim, batch_size) |> gpu;

    ps = Flux.params(generator)
    # Evaluate the loss function while calculating the pullback. We get the loss for free
    # by manually calling Zygote.pullback.
    loss, back = Zygote.pullback(ps) do
        preds = discriminator(generator(noise));
        loss = Flux.Losses.binarycrossentropy(preds, 1.)
    end
    # Evaluate the pullback with a seed-gradient of 1.0 to get the gradients for
    # the parameters of the generator
    grads = back(1.0f0)
    Flux.update!(opt_gen, Flux.params(generator), grads)
    return loss
end






end # module
