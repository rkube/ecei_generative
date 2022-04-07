### A Pluto.jl notebook ###
# v0.18.4

using Markdown
using InteractiveUtils

# ╔═╡ f533c82f-a09c-47ec-8387-50ce6efbc943
begin
	import Pkg
	Pkg.activate("/home/rkube/repos/ecei_generative")
	Pkg.instantiate()

	using ecei_generative
end

# ╔═╡ 51ca7818-6caf-11ec-341c-79ded0af6756
begin
	using Plots
	using Printf
	using StatsBase
end

# ╔═╡ 7d8169b8-5838-41d5-be00-c508676df9e2


# ╔═╡ 2dc5b778-579c-47dd-833a-195d2203d4f1
begin
	t_start = 5.0
	t_end = 7.0
	filter_f0 = 35000
	filter_f1 = 50000
	shotnr = 25878
	dev = "GR"
	datadir = @sprintf "/home/rkube/gpfs/KSTAR/%06d" shotnr
end

# ╔═╡ 3040e530-2c83-4e19-afa1-a7923e725f9f
data_filt = load_from_hdf(t_start, t_end, filter_f0, filter_f1, datadir, shotnr, dev);

# ╔═╡ ac899c28-a8b5-4122-a093-b1f54589cfce
begin
	plot(data_filt[12, 6, :])
	plot!(data_filt[12, 7, :])
	plot!(data_filt[12, 8, :])
end

# ╔═╡ 43a7fb9b-a5ff-42c6-91e1-826163fe0f74
sum(isnan.(data_filt))

# ╔═╡ 07051bac-dca7-4d25-a10b-07c7ad08f98f
data_filt[isnan.(data_filt)] .= 0f0

# ╔═╡ 4ee73e71-b92a-477a-81f5-1e33b56a1608
begin
	dt = 2e-6
	mode_t0 = t_start + 1e-3
	mode_t1 = t_start + 2e-3

	frame_0 = round((mode_t0 - t_start) / dt) |> Int;
	frame_1 = round((mode_t1 - t_start) / dt) |> Int;
end

# ╔═╡ 5193ed87-e280-41b4-952c-fd4fe83ce1b2
contourf(data_filt[:,:, frame_0 + 28], clims=(-0.075,0.075), 
	color=:bluesreds,
	xlim=(1,8),
	aspect_ratio=1)

# ╔═╡ d91f1d9b-7269-4225-ad94-3853831fe617
begin
	ftime = mode_t0
	anim = @animate for frame ∈ frame_0:frame_1

		title_str = @sprintf "%5d %s t=%8.6fs" shotnr dev ftime
		contourf(data_filt[:,:,frame], clims=(-0.075,0.075), 
			color=:bluesreds,
			aspect_ratio=1,
			title=title_str)
		ftime += dt
	end
	fname = @sprintf "%06d.gif" shotnr
	gif(anim, fname, fps=5)
end

# ╔═╡ e6a05812-cfdd-410e-acab-35d672b125f7
begin
	# Plot data normalization and look at clipping
	data_tr = clamp.(data_filt, -0.15, 0.15);
	tr = fit(UnitRangeTransform, data_tr[:]);
	data_unif = StatsBase.transform(tr, data_tr[:]);

	tr = fit(ZScoreTransform, data_tr[:]);
	data_std = StatsBase.transform(tr, data_tr[:]);
end

# ╔═╡ 58ee8996-ba95-4d19-af5a-1c6e370be491
histogram(data_unif[:])


# ╔═╡ dadd203d-c89a-4a89-9e59-052b33269fb3
histogram(data_std[:])

# ╔═╡ Cell order:
# ╠═51ca7818-6caf-11ec-341c-79ded0af6756
# ╠═f533c82f-a09c-47ec-8387-50ce6efbc943
# ╠═7d8169b8-5838-41d5-be00-c508676df9e2
# ╠═2dc5b778-579c-47dd-833a-195d2203d4f1
# ╠═3040e530-2c83-4e19-afa1-a7923e725f9f
# ╠═ac899c28-a8b5-4122-a093-b1f54589cfce
# ╠═43a7fb9b-a5ff-42c6-91e1-826163fe0f74
# ╠═07051bac-dca7-4d25-a10b-07c7ad08f98f
# ╠═4ee73e71-b92a-477a-81f5-1e33b56a1608
# ╠═5193ed87-e280-41b4-952c-fd4fe83ce1b2
# ╠═d91f1d9b-7269-4225-ad94-3853831fe617
# ╠═e6a05812-cfdd-410e-acab-35d672b125f7
# ╠═58ee8996-ba95-4d19-af5a-1c6e370be491
# ╠═dadd203d-c89a-4a89-9e59-052b33269fb3
