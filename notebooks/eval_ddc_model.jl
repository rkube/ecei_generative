### A Pluto.jl notebook ###
# v0.18.4

using Markdown
using InteractiveUtils

# ╔═╡ d5a1c146-b294-11ec-3a2f-2f1c85de9f17
begin
	import Pkg
	Pkg.activate("/home/rkube/repos/ecei_generative")
	Pkg.instantiate()
	#Pkg.add("Plots")
	#Pkg.add("BSON")

	using ecei_generative
	using Plots
	using Printf
end

# ╔═╡ 66278d63-5f53-45a1-ad14-3018c28dff7b
begin
	using Flux
	using Flux.Data: DataLoader
	using Flux: onehot, onecold, onehotbatch

	using Random
	using StatsBase
	using LinearAlgebra
	using Logging
	# using JSON
	using BSON: @load
end

# ╔═╡ fcd5a0ef-839a-4eab-a17a-cbbb9f7619d3
begin
	Core.eval(Main, :(import Flux))
	Core.eval(Main, :(import ecei_generative))
	Core.eval(Main, :(import NNlib))
end

# ╔═╡ 4ef20265-cd6e-4df7-9d82-8bc865160e9d
193*2.0/3.0

# ╔═╡ 763d9fe2-df79-4ff9-9bda-ee7a17c61d2b
args = Dict("num_depth" => 10, "batch_size" => 256, "num_classes" => 3)

# ╔═╡ 839e037c-04a9-4fce-9484-1a6f1fe71972
begin
	data_1 = load_from_hdf(2.6, 2.7, 35000, 50000, "/home/rkube/gpfs/KSTAR/025259", 25259, "GT"); # 3/2 MI
	data_2 = load_from_hdf(2.6, 2.7, 5000, 9000, "/home/rkube/gpfs/KSTAR/022289", 22289, "GT"); # ELM
	data_3 = load_from_hdf(5.0, 5.1, 5000, 9000, "/home/rkube/gpfs/KSTAR/025880", 25880, "GR"); # 2/1 MI
	data_4 = load_from_hdf(5.9, 6.0, 35000, 50000, "/home/rkube/gpfs/KSTAR/025260", 25260, "GT"); # Another 3/2 MI, not trained on
	
	0.0
end

# ╔═╡ 39ebfb16-1783-4cf6-90c9-1d9cf7a49b17
begin
	# Re-order data_1 and data_2 to have multiple channels per example
	num_samples = size(data_1)[end] ÷ args["num_depth"];
	data_1_tr = data_1[:, :, 1:num_samples * args["num_depth"]];
	clamp!(data_1_tr, -0.15, 0.15);
	trf = fit(ZScoreTransform, data_1_tr[:]);
	data_1_tr = StatsBase.transform(trf, data_1_tr[:]);
	data_1_tr = reshape(data_1_tr, (24, 8, args["num_depth"], 1, num_samples));


	num_samples = size(data_2)[end] ÷ args["num_depth"];
	data_2_tr = data_2[:, :, 1:num_samples * args["num_depth"]];
	clamp!(data_2_tr, -0.15, 0.15);
	trf = fit(ZScoreTransform, data_2_tr[:]);
	data_2_tr = StatsBase.transform(trf, data_2_tr[:]);
	data_2_tr = reshape(data_2_tr, (24, 8, args["num_depth"], 1, num_samples));

	num_samples = size(data_3)[end] ÷ args["num_depth"];
	data_3_tr = data_3[:, :, 1:num_samples * args["num_depth"]];
	clamp!(data_3_tr, -0.15, 0.15);
	trf = fit(ZScoreTransform, data_3_tr[:]);
	data_3_tr = StatsBase.transform(trf, data_3_tr[:]);
	data_3_tr = reshape(data_3_tr, (24, 8, args["num_depth"], 1, num_samples));

	num_samples = size(data_4)[end] ÷ args["num_depth"];
	data_4_tr = data_4[:, :, 1:num_samples * args["num_depth"]];
	clamp!(data_4_tr, -0.15, 0.15);
	trf = fit(ZScoreTransform, data_4_tr[:]);
	data_4_tr = StatsBase.transform(trf, data_4_tr[:]);
	data_4_tr = reshape(data_4_tr, (24, 8, args["num_depth"], 1, num_samples));

	data_all = cat(data_1_tr, data_2_tr, data_3_tr, data_4_tr, dims=5);


	# Label the various classes
	labels_1 = onehotbatch(repeat([:a], size(data_1_tr)[end]), [:a, :b, :c]);
	labels_2 = onehotbatch(repeat([:b], size(data_2_tr)[end]), [:a, :b, :c]);
	labels_3 = onehotbatch(repeat([:a], size(data_3_tr)[end]), [:a, :b, :c]);
	labels_4 = onehotbatch(repeat([:a], size(data_3_tr)[end]), [:a, :b, :c]);
	labels_all = cat(labels_1, labels_2, labels_3, labels_4, dims=2);
	
	loader_1 = DataLoader(data_1_tr, batchsize=args["batch_size"], shuffle=true);
	loader_2 = DataLoader(data_2_tr, batchsize=args["batch_size"], shuffle=true);
	loader_3 = DataLoader(data_3_tr, batchsize=args["batch_size"], shuffle=true);
	loader_4 = DataLoader(data_4_tr, batchsize=args["batch_size"], shuffle=true);
	loader_all = DataLoader((data_all, labels_all), batchsize=args["batch_size"], shuffle=true);
	0.0
end

# ╔═╡ 91f80a6d-8f55-4456-b7fd-a12abfa61e95
begin
	histogram(data_1_tr[:], xlim=[-1:1])
	histogram!(data_2_tr[:], xlims=[-5:5])
	histogram!(data_3_tr[:], xlims=[-5:5])
	histogram!(data_4_tr[:], xlims=[-5:5])
end

# ╔═╡ 727ad65e-cc1b-4c70-bf50-013ca74d4cc9
sum(abs.(data_1) .< 1e-8) / prod(size(data_1))

# ╔═╡ 5fc434cc-a7e8-4705-991d-91d60a750f70
sum(abs.(data_2) .< 1e-8) / prod(size(data_2))

# ╔═╡ 71110d16-5fa3-40ae-a122-ca3afd219049
sum(abs.(data_3) .< 1e-8) / prod(size(data_3))

# ╔═╡ 10f27c75-4e07-445b-836f-30b5c05f8f72
sum(abs.(data_4) .< 1e-8) / prod(size(data_4))

# ╔═╡ d6852a5c-89a9-4c69-873b-90f7c8b4c45d
size(data_1_tr), size(data_2_tr), size(data_3_tr),  size(data_all)

# ╔═╡ 351a0c2e-b231-4d57-b60e-c35574bd6a67
begin
	# Load a trained model from filesystem. It's on gpfs to exchange with traverse
	@load "/home/rkube/gpfs/model.bson" model_c
	# Get samples from the loader
	x1 = first(loader_1);
	x2 = first(loader_2);
	x3 = first(loader_3);
	x4 = first(loader_4);
	# Predict class assignemnts for data from every class
	cat_idx_1 = onecold(model_c(x1)[end - args["num_classes"] + 1:end, :])
	cat_idx_2 = onecold(model_c(x2)[end - args["num_classes"] + 1:end, :])
	cat_idx_3 = onecold(model_c(x3)[end - args["num_classes"] + 1:end, :])
	cat_idx_4 = onecold(model_c(x4)[end - args["num_classes"] + 1:end, :])
end

# ╔═╡ 89755852-cf6c-4ead-8250-38fafac37dac
begin
	idx = 1
	contourf(reshape(x1[:, :, :, 1, idx], (24, 8 * args["num_depth"])), title="1 -> Class $(cat_idx_1[idx])")
end

# ╔═╡ d8fa2667-7298-4942-93f5-614efc270552
begin
	contourf(reshape(x2[:, :, :, 1, idx], (24, 8 * args["num_depth"])), title="2 -> Class $(cat_idx_2[idx])")
end

# ╔═╡ 93d0f66a-7d6c-4bc5-a430-1ce0329b3949
begin
	contourf(reshape(x3[:, :, :, 1, idx], (24, 8 * args["num_depth"])), title="3 -> Class $(cat_idx_3[idx])")
end

# ╔═╡ 60be7dbf-28d5-4c16-8672-fe30a3eba16f


# ╔═╡ 22576c6e-c64a-473c-b1c6-4078e5015fee
sum(cat_idx_1 .== 1), sum(cat_idx_1 .== 2), sum(cat_idx_1 .== 3)

# ╔═╡ 5705790f-ced4-42eb-9bdf-786af2424a90
sum(cat_idx_2 .== 1), sum(cat_idx_2 .== 2), sum(cat_idx_2 .== 3)

# ╔═╡ 0f0d0f2a-30fa-42a1-bd35-aa82b2ff43ce
sum(cat_idx_3 .== 1), sum(cat_idx_3 .== 2), sum(cat_idx_3 .== 3)

# ╔═╡ ccce2023-764d-4a57-b63d-b6e4e4c172b5
sum(cat_idx_4 .== 1), sum(cat_idx_4 .== 2), sum(cat_idx_4 .== 3)

# ╔═╡ 85d377f4-e274-499e-8979-b5c2f411103c
begin
	contourf(reshape(x4[:, :, :, 1, idx], (24, 8 * args["num_depth"])), title="4 -> Class $(cat_idx_4[idx])")
end

# ╔═╡ 66450913-c2ad-43aa-aed8-8b46a032a338


# ╔═╡ Cell order:
# ╠═d5a1c146-b294-11ec-3a2f-2f1c85de9f17
# ╠═fcd5a0ef-839a-4eab-a17a-cbbb9f7619d3
# ╠═66278d63-5f53-45a1-ad14-3018c28dff7b
# ╠═4ef20265-cd6e-4df7-9d82-8bc865160e9d
# ╠═763d9fe2-df79-4ff9-9bda-ee7a17c61d2b
# ╠═839e037c-04a9-4fce-9484-1a6f1fe71972
# ╠═39ebfb16-1783-4cf6-90c9-1d9cf7a49b17
# ╠═91f80a6d-8f55-4456-b7fd-a12abfa61e95
# ╠═727ad65e-cc1b-4c70-bf50-013ca74d4cc9
# ╠═5fc434cc-a7e8-4705-991d-91d60a750f70
# ╠═71110d16-5fa3-40ae-a122-ca3afd219049
# ╠═10f27c75-4e07-445b-836f-30b5c05f8f72
# ╠═d6852a5c-89a9-4c69-873b-90f7c8b4c45d
# ╠═351a0c2e-b231-4d57-b60e-c35574bd6a67
# ╠═89755852-cf6c-4ead-8250-38fafac37dac
# ╠═d8fa2667-7298-4942-93f5-614efc270552
# ╠═93d0f66a-7d6c-4bc5-a430-1ce0329b3949
# ╠═60be7dbf-28d5-4c16-8672-fe30a3eba16f
# ╠═22576c6e-c64a-473c-b1c6-4078e5015fee
# ╠═5705790f-ced4-42eb-9bdf-786af2424a90
# ╠═0f0d0f2a-30fa-42a1-bd35-aa82b2ff43ce
# ╠═ccce2023-764d-4a57-b63d-b6e4e4c172b5
# ╠═85d377f4-e274-499e-8979-b5c2f411103c
# ╠═66450913-c2ad-43aa-aed8-8b46a032a338
