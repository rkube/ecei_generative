### A Pluto.jl notebook ###
# v0.19.9

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
	t_start = 4.0
	t_end = 5.0
	filter_f0 = 20000
	filter_f1 = 40000
	shotnr = 25086
	dev = "GT"
	datadir = @sprintf "/home/rkube/gpfs/kstar_ecei/%06d" shotnr
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
	mode_t0 = t_start
	mode_t1 = t_start + 1e-3
	dt = 2e-6

	frame_0 = ceil((mode_t0 - t_start + eps(typeof(t_start))) / dt) |> Int
	frame_1 = ceil((mode_t1 - t_start) / dt) |> Int
end

# ╔═╡ 4e5c5555-b3c2-429d-991b-58c1dcbab5d2
mode_t1

# ╔═╡ c841e1b7-51ea-409d-b381-726ca3cf2cea
frame_1 - frame_0

# ╔═╡ 5193ed87-e280-41b4-952c-fd4fe83ce1b2
contourf(data_filt[:,:, frame_0 + 18], clims=(-0.075,0.075), 
	color=:bluesreds,
	xlim=(1,8),
	aspect_ratio=1)

# ╔═╡ 70aa0c61-21d9-42d2-94f4-c9285725c3f1
size(data_filt)

# ╔═╡ 0b012e7e-1bee-4d6d-a610-55646ac71b32
frame_0

# ╔═╡ 4ef32f94-a09a-4ac1-80bf-ccfa307dc9d5
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
	fname_anim = @sprintf "%06d_reorder.gif" shotnr
	gif(anim, fname_anim, fps=5)
end

# ╔═╡ d91f1d9b-7269-4225-ad94-3853831fe617


begin
	# Plot data normalization and look at clipping
	data_tr = clamp.(data_filt, -0.15, 0.15);
	tr = fit(UnitRangeTransform, data_tr[:]);
	data_unif = StatsBase.transform(tr, data_tr[:]);

	tr = fit(ZScoreTransform, data_tr[:]);
	data_std = StatsBase.transform(tr, data_tr[:]);
end

# ╔═╡ f406951c-be9b-4262-b724-788b5f025a7c
let
	p = histogram(data_filt[:], title="Shot $(shotnr) - Processed")
	fname = @sprintf "%06d_hist_processed.png" shotnr
	savefig(p, fname)
end

# ╔═╡ c82d1d8f-0a82-478e-8de0-b60c264a76f6
let
	p = histogram(data_unif[:], title="Shot $(shotnr) - UnitRangeTransform")
	fname = @sprintf "%06d_hist_unitrg.png" shotnr
	savefig(p, fname)
end

# ╔═╡ 6531f024-ff36-4682-82ce-9f9c57482919
let
	p = histogram(data_std[:], title="Shot $(shotnr) - ZScoreTransform")
	fname = @sprintf "%06d_hist_zscore.png" shotnr
	savefig(p, fname)
end

# ╔═╡ Cell order:
# ╠═51ca7818-6caf-11ec-341c-79ded0af6756
# ╠═f533c82f-a09c-47ec-8387-50ce6efbc943
# ╠═7d8169b8-5838-41d5-be00-c508676df9e2
# ╠═2dc5b778-579c-47dd-833a-195d2203d4f1
# ╠═3040e530-2c83-4e19-afa1-a7923e725f9f
# ╠═ac899c28-a8b5-4122-a093-b1f54589cfce
# ╠═4ee73e71-b92a-477a-81f5-1e33b56a1608
# ╠═4e5c5555-b3c2-429d-991b-58c1dcbab5d2
# ╠═c841e1b7-51ea-409d-b381-726ca3cf2cea
# ╠═5193ed87-e280-41b4-952c-fd4fe83ce1b2
# ╠═70aa0c61-21d9-42d2-94f4-c9285725c3f1
# ╠═0b012e7e-1bee-4d6d-a610-55646ac71b32
# ╠═4ef32f94-a09a-4ac1-80bf-ccfa307dc9d5
# ╠═d91f1d9b-7269-4225-ad94-3853831fe617
# ╠═f406951c-be9b-4262-b724-788b5f025a7c
# ╠═c82d1d8f-0a82-478e-8de0-b60c264a76f6
# ╠═6531f024-ff36-4682-82ce-9f9c57482919
