### A Pluto.jl notebook ###
# v0.18.4

using Markdown
using InteractiveUtils

# ╔═╡ b043daa2-b1d5-11ec-2c06-9f78c063ae7c
begin
	import Pkg
	Pkg.activate("/home/rkube/repos/ecei_generative")
	Pkg.instantiate()
	Pkg.add("Plots")

	using ecei_generative
	using Plots
	using Printf
	using Flux: onehotbatch
	using Random
end

# ╔═╡ a2f2413c-8192-47a6-9d01-d5d16aedbeab
using Flux.Data: DataLoader

# ╔═╡ 38307b8a-7c74-44c7-a423-ab26421cae19
md"# Test dataloaders

Construct the dataloader used in ML training, retrieve single samples and plot them. 
"

# ╔═╡ 385ef332-8baf-4dc1-ad49-9278bf92078b
args = Dict("num_depth" => 5, "batch_size" => 64)

# ╔═╡ 07491312-8c69-43f3-8733-93d6a20e0d4f
begin
	data_1 = load_from_hdf(2.6, 2.7, 35000, 50000, "/home/rkube/gpfs/KSTAR/025259", 25259, "GT");
	data_2 = load_from_hdf(2.6, 2.7, 5000, 9000, "/home/rkube/gpfs/KSTAR/022289", 22289, "GT");
end

# ╔═╡ e4b716e4-f722-4684-8d3a-0523a0814b79
begin
	# Re-order data_1 and data_2 to have multiple channels per example
	num_samples_1 = size(data_1)[end] ÷ args["num_depth"];
	data1_tr = reshape(data_1[:, :, 1:num_samples_1 * args["num_depth"]], (24, 8, args["num_depth"], num_samples_1));
	
	num_samples_2 = size(data_2)[end] ÷ args["num_depth"];
	data2_tr = reshape(data_2[:, :, 1:num_samples_2 * args["num_depth"]], (24, 8, args["num_depth"], num_samples_2));
	
	data_all = cat(data1_tr, data2_tr, dims=4);
	data_all = reshape(data_all, (size(data_all)[1], size(data_all)[2], size(data_all)[3], 1, size(data_all)[end]));
	
	# # Scale data to [-1.0; 1.0]
	data_all = 2.0 * (data_all .- minimum(data_all)) / (maximum(data_all) - minimum(data_all)) .- 1.0; 
	
	# # Label the various classes
	labels_1 = onehotbatch(repeat([:a], size(data1_tr)[4]), [:a, :b])
	labels_2 = onehotbatch(repeat([:b], size(data2_tr)[4]), [:a, :b])
	labels_all = cat(labels_1, labels_2, dims=2);
	
	# # Train / test split
	split_ratio = 0.8 
	num_samples = size(data_all)[5]
	num_train = round(split_ratio * num_samples) |> Int 
	idx_all = randperm(num_samples);      # Random indices for all samples
	idx_train = idx_all[1:num_train];     # Indices for training set
	idx_test = idx_all[num_train:end];    # Indices for test set
end

# ╔═╡ 60cc62d1-3a3c-4748-8195-211c230d71e6
loader_train = DataLoader((data_all[:, :, :, :, idx_train], labels_all[:, idx_train]), batchsize=args["batch_size"], shuffle=true);
d

# ╔═╡ ff3b89bf-9cf3-4099-a6f0-db240460101e
(x, y) = first(loader_train);

# ╔═╡ b10c1748-3434-458d-9cf2-e1344fa44d66


# ╔═╡ 13dee42d-34d2-4de4-b9c3-89149401c797
# plot([contourf(x[:,:,i,1,1]) for i ∈ 1:5]..., layout=(5, 1))

# ╔═╡ bfde6d05-6c46-4c65-bfa7-199d2f02da0b
contourf(x[:,:,1,1,1])

# ╔═╡ f3ada9c1-523f-4562-8494-ab019915815b
contourf(x[:,:,2,1,1])

# ╔═╡ 7e70d603-220e-41f9-94df-5a74a39960a5
contourf(x[:,:,3,1,1])

# ╔═╡ 07c057f1-f6c5-48eb-b31e-1f6c9293bc0b
contourf(x[:,:,4,1,1])

# ╔═╡ a2753067-7912-4ee2-bb8c-144b670d9fc6
contourf(x[:,:,5,1,1])

# ╔═╡ Cell order:
# ╠═38307b8a-7c74-44c7-a423-ab26421cae19
# ╠═b043daa2-b1d5-11ec-2c06-9f78c063ae7c
# ╠═a2f2413c-8192-47a6-9d01-d5d16aedbeab
# ╠═385ef332-8baf-4dc1-ad49-9278bf92078b
# ╠═07491312-8c69-43f3-8733-93d6a20e0d4f
# ╠═e4b716e4-f722-4684-8d3a-0523a0814b79
# ╠═60cc62d1-3a3c-4748-8195-211c230d71e6
# ╠═ff3b89bf-9cf3-4099-a6f0-db240460101e
# ╠═b10c1748-3434-458d-9cf2-e1344fa44d66
# ╠═13dee42d-34d2-4de4-b9c3-89149401c797
# ╠═bfde6d05-6c46-4c65-bfa7-199d2f02da0b
# ╠═f3ada9c1-523f-4562-8494-ab019915815b
# ╠═7e70d603-220e-41f9-94df-5a74a39960a5
# ╠═07c057f1-f6c5-48eb-b31e-1f6c9293bc0b
# ╠═a2753067-7912-4ee2-bb8c-144b670d9fc6
