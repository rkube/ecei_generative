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

	import MultivariateStats
end

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
	push!(df, [25973, "GR", 5.754, 5.755, 1000, 20000]) # 2/1 + 3/2
	push!(df, [25978, "GR", 5.754, 5.755, 20000, 40000]) # 3/2 island
end

# ╔═╡ a4a585df-f581-40fd-8a47-95cc5d652bbe
# Labels encode the visibile island structure
labels = vcat(ones(500), ones(500), 2 * ones(500), 2 * ones(500), 2 * ones(500), 3 * ones(500), ones(500));

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
	group = 6
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

	# pratio = []
	# for ndim ∈ 1:32
	# 	M = fit(PCA, X_train; maxoutdim=ndim);
	# 	push!(pratio, principalratio(M));
	# end
	M = fit(PCA, X_train; maxoutdim=dimK);

	Y_train = predict(M, X_test);
	Y_test = predict(M, X_test);
	Y_all = hcat(Y_test, Y_train);
	X_rec = reconstruct(M, Y_test);
	M
	principalratio(M)
end

# ╔═╡ 1a7ee0bb-5d47-4cb2-856b-871e9341c8e5
# This data is the PCA principal ratio as a function of dimensions. Evaluated in the cell above.

pratio = [0.290261, 0.534017, 0.619, 0.694753, 0.762177, 0.814929, 0.854322, 0.881777, 0.89983, 0.91654, 0.928815, 0.936205, 0.941482, 0.946276, 0.950536, 0.954209, 0.957313, 0.960311, 0.962968, 0.965479, 0.967567, 0.969571, 0.971212, 0.972696, 0.974053, 0.9753, 0.976326, 0.97732, 0.978233, 0.979031, 0.979759, 0.980442]

# ╔═╡ 654213a4-dcb0-48e1-9ba1-74b8ac7c8472
plot(pratio)

# ╔═╡ 604f4803-5c2b-4763-a8a8-7e3857d92813
begin
	frame = 150
	plot(contourf(reshape(X_rec[:, frame], (24,8))),
		 contourf(reshape(X_test[:, frame], (24,8))))
end

# ╔═╡ 9f156f47-b8c8-4ed1-bfaf-f658c88613fc
# K Means is a statistical algorithm. We should evaluate this cell a couple of times and see how robust it performs against the true labels.
# In reality there are 3 classes of islands.

R = kmeans(Y_all, 3; maxiter=100, display=:iter);

# ╔═╡ 23181019-b5e8-4dfe-b38e-031e11064043
begin
	plot(assignments(R), label="K-means")
	plot!(labels, label="True")
end

# ╔═╡ Cell order:
# ╠═e0d69380-6d6a-11ec-237a-699b55edf589
# ╠═5237a520-bf33-4f6b-839a-9c2fda96058f
# ╠═23a5eb2e-0b9d-43b5-a880-8a36973ae7ee
# ╠═ca8ec936-2840-41ad-baf8-a1e13616beba
# ╠═8a4f7b48-bd68-4bdc-9063-bc520dc73f39
# ╠═a4a585df-f581-40fd-8a47-95cc5d652bbe
# ╠═3a46588f-9b04-4697-91b8-13884f01413c
# ╠═019795d6-c1f5-4853-b594-c042af309787
# ╠═fca588b2-db1b-4e40-acfc-cb6f960f10a4
# ╟─aa4681ce-758d-416f-9407-110dcfdef8c8
# ╠═12960ffb-1c56-409e-8bf7-3867c99a8176
# ╠═eeb33e04-32ba-4c66-a7d1-e931a3cef6ba
# ╠═c7a4d9ee-e329-4a50-86c1-3131a08f3aa5
# ╟─4cf05447-fb76-417c-b1f5-17f9112920a3
# ╠═83700348-b0ea-45ca-85e4-2342c2876249
# ╠═1a7ee0bb-5d47-4cb2-856b-871e9341c8e5
# ╠═654213a4-dcb0-48e1-9ba1-74b8ac7c8472
# ╠═604f4803-5c2b-4763-a8a8-7e3857d92813
# ╠═9f156f47-b8c8-4ed1-bfaf-f658c88613fc
# ╠═23181019-b5e8-4dfe-b38e-031e11064043
