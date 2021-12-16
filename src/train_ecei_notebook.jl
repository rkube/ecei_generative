### A Pluto.jl notebook ###
# v0.16.0

using Markdown
using InteractiveUtils

# ╔═╡ 15ed20c6-3284-11ec-2d4f-7bb1343debcf
begin
	using Pkg
	Pkg.activate("..")
end


# ╔═╡ 2c62d1d7-efa3-4656-9777-987721d5cffe
begin
	using ecei_generative: get_framedata, train_dscr!, train_gen!
	using Flux
	using Flux.Data: DataLoader
	using Zygote
	using Printf
	using Plots
end

# ╔═╡ c6138d52-8ad8-4e07-8041-2acc215cbf9d
begin
	n_features = 24*8
	latent_dim = 64
	batch_size = 512
	output_period = 100 
	num_epochs = 50
	opt_dscr = ADAM(2e-4)
	opt_gen = ADAM(2e-4)
end

# ╔═╡ 9cac3228-625c-4eaf-b399-59699742de09
begin
	data_137 = get_framedata(25259, "GT", 137, "/home/rkube/repos/ecei_generative/data")
	data_138 = get_framedata(25259, "GT", 138, "/home/rkube/repos/ecei_generative/data")
	data_139 = get_framedata(25259, "GT", 139, "/home/rkube/repos/ecei_generative/data")
end

# ╔═╡ 60886e18-b47c-4e66-93d8-86b93c708ac0
begin
	data = cat(data_137, data_138, data_139, dims=3) |> gpu
	train_loader = DataLoader(data, batchsize=batch_size, shuffle=true);
	nothing
end

# ╔═╡ 764e1b4e-06bf-4614-bf40-1e81e2cc98ab
begin
	discriminator = Chain(Dense(n_features, 1024, x -> leakyrelu(x, 0.2f0)),
						  Dropout(0.3),
						  Dense(1024, 512, x -> leakyrelu(x, 0.2f0)),
						  Dropout(0.3),
						  Dense(512, 256, x -> leakyrelu(x, 0.2f0)),
						  Dropout(0.3),
						  Dense(256, 1, sigmoid)) |> gpu;

	generator = Chain(Dense(latent_dim, 256, x -> leakyrelu(x, 0.2f0)),
					  Dense(256, 512, x -> leakyrelu(x, 0.2f0)),
					  Dense(512, 1024, x -> leakyrelu(x, 0.2f0)),
					  Dense(1024, n_features, tanh)) |> gpu;
end

# ╔═╡ a06590b6-54d2-436c-b3a9-f83d1dda280f


# ╔═╡ d2559c3b-3f88-4ecf-bda1-0bf2da3f5cda
begin
	# Loss vectors
	lossvec_g = zeros(num_epochs)
	lossvec_d = zeros(num_epochs)

	# Main training loop
	for n ∈ 1:num_epochs
		println(n)
		Σ_loss_g = 0.0f0
		Σ_loss_d = 0.0f0

		for x in train_loader
			this_batch = size(x)[end]
			real_data = flatten(x)

			# Generate noise
			noise = randn(latent_dim, this_batch) |> gpu 
			fake_data = generator(noise)
			Σ_loss_d += train_dscr!(discriminator, real_data, fake_data, this_batch, opt_dscr)
			Σ_loss_g = train_gen!(discriminator, generator, opt_gen, latent_dim, batch_size)
		end 

		lossvec_d[n] = Σ_loss_d / size(data)[end]
		lossvec_g[n] = Σ_loss_g / size(data)[end]
	end
end

# ╔═╡ 45f34792-1909-40ec-b102-255b2d150d7f
lossvec_g

# ╔═╡ 2c218b2e-55ac-46aa-b59f-03ced84e9883
begin
	num_samples = 4
	noise = randn(Float32, latent_dim, num_samples) |> gpu
	fake_data = reshape(generator(noise), 24, 8, num_samples) |> cpu
	
	contour_list = [contourf(fake_data[:, :, i]) for i ∈ 1:num_samples]
	p = plot(contour_list..., layout=(2,2))
end

# ╔═╡ 62aafb0b-916f-4fa0-afb1-b3f7585809fd
savefig(p, "GAN_trained_1000epochs_ecei.png")

# ╔═╡ 7cac0e61-5315-47a3-914c-cbecb25e6b26
discriminator(flatten(x))

# ╔═╡ d2e22655-261f-4971-a355-43362668fa5f
let
	noise = randn(Float32, latent_dim, 4) |> gpu
	fake_data = generator(noise);
	preds = discriminator(fake_data);
end

# ╔═╡ Cell order:
# ╠═15ed20c6-3284-11ec-2d4f-7bb1343debcf
# ╠═2c62d1d7-efa3-4656-9777-987721d5cffe
# ╠═c6138d52-8ad8-4e07-8041-2acc215cbf9d
# ╠═9cac3228-625c-4eaf-b399-59699742de09
# ╠═60886e18-b47c-4e66-93d8-86b93c708ac0
# ╠═764e1b4e-06bf-4614-bf40-1e81e2cc98ab
# ╠═a06590b6-54d2-436c-b3a9-f83d1dda280f
# ╠═d2559c3b-3f88-4ecf-bda1-0bf2da3f5cda
# ╠═45f34792-1909-40ec-b102-255b2d150d7f
# ╠═2c218b2e-55ac-46aa-b59f-03ced84e9883
# ╠═62aafb0b-916f-4fa0-afb1-b3f7585809fd
# ╠═7cac0e61-5315-47a3-914c-cbecb25e6b26
# ╠═d2e22655-261f-4971-a355-43362668fa5f
