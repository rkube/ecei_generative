# Utility function

export plot_fake

using Plots
using Printf

function plot_fake(G, args, epoch; num_samples=5)
   
    noise = randn(args["latent_dim"], num_samples) |> gpu;
    fake_x = G(noise) |> cpu;
    for sample in 1:num_samples
        p = contourf(fake_x[:,:,1,sample]', aspectratio=1)
        fname = @sprintf "epoch_%03d_sample_%02d.png" epoch sample
        savefig(p, fname)
    end

    p = contourf(fake_x[:,:,1,1]', aspectratio=1)

end


