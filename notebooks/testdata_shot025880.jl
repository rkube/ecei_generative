### A Pluto.jl notebook ###
# v0.18.4

using Markdown
using InteractiveUtils

# ╔═╡ f533c82f-a09c-47ec-8387-50ce6efbc943
begin
	import Pkg
	Pkg.activate("/home/rkube/repos/ecei_generative")
	Pkg.instantiate()
	# Pkg.add("Plots")

	using ecei_generative
end

# ╔═╡ 51ca7818-6caf-11ec-341c-79ded0af6756
begin
	using Plots
	using Printf
	using Statistics
end

# ╔═╡ 7d8169b8-5838-41d5-be00-c508676df9e2


# ╔═╡ 2dc5b778-579c-47dd-833a-195d2203d4f1
begin
	t_start = 5.0
	t_end = 6.0
	filter_f0 = 5000
	filter_f1 = 9000
	shotnr = 25880
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

# ╔═╡ 4ee73e71-b92a-477a-81f5-1e33b56a1608
begin
	dt = 2e-6
	mode_t0 = 5.016
	mode_t1 = 5.017
	
	frame_0 = convert(Int, round((mode_t0 - t_start) / dt)) 
	frame_1 = convert(Int, round((mode_t1 - t_start) / dt))
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
# 	fname = @sprintf "%06d_reorder.gif" shotnr
# 	gif(anim, fname, fps=5)
# end

# ╔═╡ 7267af48-9d79-4dd3-aa74-38423dfa0fe8
# Look at the standardized histograms

# ╔═╡ fe2567c4-a4e5-4e37-bb2a-656e806fffd8
begin
	# Plot data normalization and look at clipping
	data_tr = clamp.(data_filt, -0.15, 0.15);
	tr = fit(UnitRangeTransform, data_tr[:]);
	data_unif = StatsBase.transform(tr, data_tr[:]);

	tr = fit(ZScoreTransform, data_tr[:]);
	data_std = StatsBase.transform(tr, data_tr[:]);
end

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

# ╔═╡ Cell order:
# ╠═51ca7818-6caf-11ec-341c-79ded0af6756
# ╠═f533c82f-a09c-47ec-8387-50ce6efbc943
# ╠═7d8169b8-5838-41d5-be00-c508676df9e2
# ╠═2dc5b778-579c-47dd-833a-195d2203d4f1
# ╠═3040e530-2c83-4e19-afa1-a7923e725f9f
# ╠═ac899c28-a8b5-4122-a093-b1f54589cfce
# ╠═4ee73e71-b92a-477a-81f5-1e33b56a1608
# ╠═c841e1b7-51ea-409d-b381-726ca3cf2cea
# ╠═5193ed87-e280-41b4-952c-fd4fe83ce1b2
# ╠═d91f1d9b-7269-4225-ad94-3853831fe617
# ╠═7267af48-9d79-4dd3-aa74-38423dfa0fe8
# ╠═fe2567c4-a4e5-4e37-bb2a-656e806fffd8
# ╠═6f651a6a-4781-4aab-9eed-484b0eef449a
# ╠═3c4f0962-6f07-47ed-a1e0-d818afa5be42
# ╠═25f4477d-7494-4e99-9fcb-43517d88f15c
