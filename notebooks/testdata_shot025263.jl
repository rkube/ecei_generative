### A Pluto.jl notebook ###
# v0.18.1

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
	t_start = 2.0
	t_end = 3.0
	filter_f0 = 10000
	filter_f1 = 25000
	shotnr = 25263
	dev = "GR"
	datadir = @sprintf "/home/rkube/gpfs/KSTAR/%06d" shotnr
end

# ╔═╡ 3040e530-2c83-4e19-afa1-a7923e725f9f
data_filt = load_from_hdf(t_start, t_end, filter_f0, filter_f1, datadir, shotnr, dev);

# ╔═╡ 8144a2e5-5567-4ea9-a0a5-aef8e8a3c000
data_filt[isnan.(data_filt)] .= 0f0

# ╔═╡ ac899c28-a8b5-4122-a093-b1f54589cfce
begin
	plot(data_filt[12, 6, :])
	plot!(data_filt[12, 7, :])
	plot!(data_filt[12, 8, :])
end

# ╔═╡ 4ee73e71-b92a-477a-81f5-1e33b56a1608
begin
	dt = 2e-6
	mode_t0 = t_start
	mode_t1 = t_start + 1e-3
	
	frame_0 = round((mode_t0 - 2.0) / dt) |> Int;
	frame_1 = round((mode_t1 - 2.0) / dt) |> Int;
end

# ╔═╡ c841e1b7-51ea-409d-b381-726ca3cf2cea
frame_1 - frame_0

# ╔═╡ 5193ed87-e280-41b4-952c-fd4fe83ce1b2
contourf(data_filt[:,:, frame_0 + 18], clims=(-0.075,0.075), 
	color=:bluesreds,
	xlim=(1,8),
	aspect_ratio=1)

# ╔═╡ d91f1d9b-7269-4225-ad94-3853831fe617
# begin
# 	ftime = mode_t0
# 	anim = @animate for frame ∈ frame_0:frame_1

# 		title_str = @sprintf "%5d %s t=%8.6fs" shotnr dev ftime
# 		contourf(data_filt[:,:,frame], clims=(-0.075,0.075), 
# 			color=:bluesreds,
# 			aspect_ratio=1,
# 			title=title_str)
# 		ftime += dt
# 	end
# 	fname = @sprintf "%06d.gif" shotnr
# 	gif(anim, fname, fps=5)
# end

# ╔═╡ f406951c-be9b-4262-b724-788b5f025a7c
p = histogram(data_filt[:], title="Shot $(shotnr) - Processed")
fname = @sprintf "%06d_hist_processed.png" shotnr
savefig(p, fname)

# ╔═╡ c82d1d8f-0a82-478e-8de0-b60c264a76f6
p = histogram(data_unif[:], title="Shot $(shotnr) - UnitRangeTransform")
fname = @sprintf "%06d_hist_unitrg.png" shotnr
savefig(p, fname)

# ╔═╡ 6531f024-ff36-4682-82ce-9f9c57482919
p = histogram(data_std[:], title="Shot $(shotnr) - ZScoreTransform")
fname = @sprintf "%06d_hist_zscore.png" shotnr
savefig(p, fname)

# ╔═╡ e6a05812-cfdd-410e-acab-35d672b125f7
begin
	# Plot data normalization and look at clipping
	data_tr = clamp.(data_filt, -0.15, 0.15);
	tr = fit(UnitRangeTransform, data_tr[:]);
	data_unif = StatsBase.transform(tr, data_tr[:]);

	tr = fit(ZScoreTransform, data_tr[:]);
	data_std = StatsBase.transform(tr, data_tr[:]);
end

# ╔═╡ Cell order:
# ╠═51ca7818-6caf-11ec-341c-79ded0af6756
# ╠═f533c82f-a09c-47ec-8387-50ce6efbc943
# ╠═7d8169b8-5838-41d5-be00-c508676df9e2
# ╠═2dc5b778-579c-47dd-833a-195d2203d4f1
# ╠═3040e530-2c83-4e19-afa1-a7923e725f9f
# ╠═8144a2e5-5567-4ea9-a0a5-aef8e8a3c000
# ╠═ac899c28-a8b5-4122-a093-b1f54589cfce
# ╠═4ee73e71-b92a-477a-81f5-1e33b56a1608
# ╠═c841e1b7-51ea-409d-b381-726ca3cf2cea
# ╠═5193ed87-e280-41b4-952c-fd4fe83ce1b2
# ╠═d91f1d9b-7269-4225-ad94-3853831fe617
# ╠═f406951c-be9b-4262-b724-788b5f025a7c
# ╠═c82d1d8f-0a82-478e-8de0-b60c264a76f6
# ╠═6531f024-ff36-4682-82ce-9f9c57482919
# ╠═e6a05812-cfdd-410e-acab-35d672b125f7
