### A Pluto.jl notebook ###
# v0.17.3

using Markdown
using InteractiveUtils

# ╔═╡ e0d69380-6d6a-11ec-237a-699b55edf589
begin
	import Pkg
	Pkg.activate("/home/rkube/repos/ecei_generative")
	Pkg.instantiate()

	using ecei_generative
	using Plots
	using Printf
	using DataFrames
	using Clustering
	using MultivariateStats
	using RDatasets
	using Flux
	using EvalMetrics
	
	import MultivariateStats
end

# ╔═╡ b0a17e23-63c9-46b5-915b-0d1ecf441bac
# Pkg.add("EvalMetrics")

# ╔═╡ 5237a520-bf33-4f6b-839a-9c2fda96058f
md"""# Try clustering to separate the ECEi frames with different magnetic islands from one another."""

# ╔═╡ 23a5eb2e-0b9d-43b5-a880-8a36973ae7ee


# ╔═╡ ca8ec936-2840-41ad-baf8-a1e13616beba
# Monkey patching. See https://github.com/JuliaStats/MultivariateStats.jl/pull/167
begin
	import StatsBase.predict
	predict(M::PCA{T}, x::AbstractVecOrMat{T}) where {T<:Real} = transpose(M.proj) * centralize(x, M.mean)
end

# ╔═╡ 8a4f7b48-bd68-4bdc-9063-bc520dc73f39
# Generate a DataFrame with the parameters of the datasets
begin
	# Create test data structure
	df = DataFrame(shotnr=Int[], dev=String[], tstart=Float64[], tend=Float64[], f0=Int[], f1=[])
	push!(df, [25259, "GT", 2.659, 2.660, 35000, 50000]) # 3/2 island
	push!(df, [25260, "GT", 5.909, 5.910, 35000, 50000]) # 3/2 island
	# Dataset for 25878 has nans
	# push!(df, [25878, "GR", 3.90, 3.901, 35000, 50000])
	push!(df, [25879, "GR", 5.904, 5.905, 5000, 20000]) # 2/1 island
	push!(df, [25880, "GR", 5.404, 5.405, 1000, 20000]) # 2/1 island
	push!(df, [25897, "GT", 5.914, 5.915, 1000, 20000]) # 2/1 island
	#push!(df, [25973, "GR", 5.754, 5.755, 1000, 20000]) # 2/1 + 3/2
	push!(df, [25978, "GR", 5.754, 5.755, 20000, 40000]) # 3/2 island
end

# ╔═╡ a4a585df-f581-40fd-8a47-95cc5d652bbe
# Labels encode the visibile island structure
labels = vcat(ones(499), ones(499), 2 * ones(499), 2 * ones(499), 2 * ones(499), ones(499));

# ╔═╡ 3a46588f-9b04-4697-91b8-13884f01413c
# Fetch datasets
begin
	data_list = []
	for row in eachrow(df)
		datadir = @sprintf "/home/rkube/gpfs/KSTAR/%06d" row.shotnr
		data_filt = load_from_hdf(row.tstart, row.tend, row.f0, row.f1, datadir, row.shotnr, row.dev);
		push!(data_list, data_filt)
	end
end

# ╔═╡ 019795d6-c1f5-4853-b594-c042af309787
Z = reshape(cat(data_list..., dims=1), :, 192);

# ╔═╡ 1a3c00eb-d187-45c1-aded-bc89974750ac
size(Z)

# ╔═╡ fca588b2-db1b-4e40-acfc-cb6f960f10a4
contourf(reshape(Z[108,:], (24, 8)))

# ╔═╡ aa4681ce-758d-416f-9407-110dcfdef8c8
md"""With the data loaded, now we explore how to separate images from the different island structures from one another.

As a first step we visualize the 24x8=192 element vectors using line plots.
"""

# ╔═╡ 12960ffb-1c56-409e-8bf7-3867c99a8176
# Plot frames from first shot with 3/2 islands as vectors.
let
	group = 1
	plot(Z[group * 499 + 10, :], ylims=(-0.1, 0.1))
	plot!(Z[group * 499 + 50, :])
	plot!(Z[group * 499 + 100, :])
	plot!(Z[group * 499 + 150, :])
	plot!(Z[group * 499 + 200, :])
	plot!(Z[group * 499 + 250, :])
end

# ╔═╡ eeb33e04-32ba-4c66-a7d1-e931a3cef6ba
# Plot frames from last shot with 2/1 islands as vectors.
# See if we can visually discern them from 3/2 islands?
let
	group = 4
	plot(Z[group * 499 + 10, :], ylims=(-0.1, 0.1))
	plot!(Z[group * 499 + 50, :])
	plot!(Z[group * 499 + 100, :])
	plot!(Z[group * 499 + 150, :])
	plot!(Z[group * 499 + 200, :])
	plot!(Z[group * 499 + 250, :])
end

# ╔═╡ c7a4d9ee-e329-4a50-86c1-3131a08f3aa5
md"""Visually they look very similar. There may be more oscillations in the 3/2 image than in the 2/1 image."""

# ╔═╡ 4cf05447-fb76-417c-b1f5-17f9112920a3
md"""# Perform PCA on the images.

Clustering algorithms like k-means rely on euclidian distance. But this metric becomes useless in high dimensions. So if we want clustering to work we need to find a way for reducing the dimensions that the clustering algorithms will work with.

A first approach for this is PCA.
"""

# ╔═╡ 83700348-b0ea-45ca-85e4-2342c2876249
begin
	dimK = 16
	X_train = Matrix(Z[1:2:end, :])';
	X_test = Matrix(Z[2:2:end, :])';
	labels_test = labels[2:2:end]

	# pratio = []
	# for ndim ∈ 1:32
	# 	M = fit(PCA, X_train; maxoutdim=ndim);
	# 	push!(pratio, principalratio(M));
	# end
	M = fit(PCA, X_train; maxoutdim=dimK);

	Y_train = predict(M, X_test);
	Y_test = predict(M, X_test);
	#Y_all = hcat(Y_test, Y_train);
	X_rec = reconstruct(M, Y_test);
	M
	principalratio(M)
end

# ╔═╡ 84a3bb31-3f03-42c7-8568-4f60b804d5e0
size(X_train), size(X_test)

# ╔═╡ 3e2ab4bf-036d-41a3-988c-28edc2a87afb
begin
	pratio = []
	dim_range = 1:32
	for ndim ∈ dim_range
		M = fit(PCA, X_train; maxoutdim=ndim);
		push!(pratio, principalratio(M));
	end
end

# ╔═╡ 654213a4-dcb0-48e1-9ba1-74b8ac7c8472
let
	p = plot(dim_range, pratio, xlabel="ndim", ylabel="Principal ratio")
	savefig(p, "dim_ratio_pca.png")
	p
end

# ╔═╡ 604f4803-5c2b-4763-a8a8-7e3857d92813
begin
	frame = 150
	plot(contourf(reshape(X_rec[:, frame], (24,8))),
		 contourf(reshape(X_test[:, frame], (24,8))))
end

# ╔═╡ 9f156f47-b8c8-4ed1-bfaf-f658c88613fc
# K Means is a statistical algorithm. We should evaluate this cell a couple of times and see how robust it performs against the true labels.
# In reality there are 3 classes of islands.

R = kmeans(Y_test, 2; maxiter=100, display=:iter);

# ╔═╡ 0179a882-c80a-40b4-a3b1-6df161244c7a
unique(labels)

# ╔═╡ e1efdbd0-06cc-4845-ad66-8643fa1ba2b8
cm = ConfusionMatrix(labels_test .- 1, assignments(R) .- 1)

# ╔═╡ 9087fcdd-cf89-429a-8863-53ac6f38459d
begin
	p = heatmap([cm.tp cm.fp; cm.tn cm.fn], color_palette=cgrad(:algae))
	# annotate!(p, [(1.0, 2.0, (string("TP: ", cm.tp), 16, :white)),
	# 	          (2.0, 2.0, (string("FP: ", cm.fp), 16, :white)),
	# 	          (1.0, 1.0, (string("TN: ", cm.tn), 16, :black)),
	# 	          (2.0, 1.0, (string("FN: ", cm.fn), 16, :black))])
end



# annotate!([(5, y[5], ("this is #5", 16, :red, :center)), (10, y[10], ("this is #10", :right, 20, "courier"))])


# ╔═╡ 4e546a96-68a4-4b5e-a95a-b40f7390ec6a
Flux.Losses.binarycrossentropy(assignments(R) .- 1.0, labels_test .- 1.0) / length(labels_test)

# ╔═╡ Cell order:
# ╠═e0d69380-6d6a-11ec-237a-699b55edf589
# ╠═b0a17e23-63c9-46b5-915b-0d1ecf441bac
# ╠═5237a520-bf33-4f6b-839a-9c2fda96058f
# ╠═23a5eb2e-0b9d-43b5-a880-8a36973ae7ee
# ╠═ca8ec936-2840-41ad-baf8-a1e13616beba
# ╠═8a4f7b48-bd68-4bdc-9063-bc520dc73f39
# ╠═a4a585df-f581-40fd-8a47-95cc5d652bbe
# ╠═3a46588f-9b04-4697-91b8-13884f01413c
# ╠═1a3c00eb-d187-45c1-aded-bc89974750ac
# ╠═019795d6-c1f5-4853-b594-c042af309787
# ╠═fca588b2-db1b-4e40-acfc-cb6f960f10a4
# ╟─aa4681ce-758d-416f-9407-110dcfdef8c8
# ╠═12960ffb-1c56-409e-8bf7-3867c99a8176
# ╠═eeb33e04-32ba-4c66-a7d1-e931a3cef6ba
# ╠═c7a4d9ee-e329-4a50-86c1-3131a08f3aa5
# ╟─4cf05447-fb76-417c-b1f5-17f9112920a3
# ╠═84a3bb31-3f03-42c7-8568-4f60b804d5e0
# ╠═83700348-b0ea-45ca-85e4-2342c2876249
# ╠═3e2ab4bf-036d-41a3-988c-28edc2a87afb
# ╠═654213a4-dcb0-48e1-9ba1-74b8ac7c8472
# ╠═604f4803-5c2b-4763-a8a8-7e3857d92813
# ╠═9f156f47-b8c8-4ed1-bfaf-f658c88613fc
# ╠═0179a882-c80a-40b4-a3b1-6df161244c7a
# ╠═e1efdbd0-06cc-4845-ad66-8643fa1ba2b8
# ╠═9087fcdd-cf89-429a-8863-53ac6f38459d
# ╠═4e546a96-68a4-4b5e-a95a-b40f7390ec6a
