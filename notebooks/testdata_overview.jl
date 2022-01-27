### A Pluto.jl notebook ###
# v0.17.3

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
end

# ╔═╡ 7d8169b8-5838-41d5-be00-c508676df9e2


# ╔═╡ 2dc5b778-579c-47dd-833a-195d2203d4f1
begin
	t_start = 2.0
	t_end = 10.0
	filter_f0 = 20000
	filter_f1 = 40000
	shotnr = 25978
	dev = "GR"
	datadir = @sprintf "/home/rkube/gpfs/KSTAR/%06d" shotnr
end

# ╔═╡ 3040e530-2c83-4e19-afa1-a7923e725f9f
data_filt = load_from_hdf(t_start, t_end, filter_f0, filter_f1, datadir, shotnr, dev);

# ╔═╡ ac899c28-a8b5-4122-a093-b1f54589cfce
begin
	plot(data_filt[370_000:380_000, 12, 6])
	plot!(data_filt[370_000:380_000, 12, 7])
	plot!(data_filt[370_000:380_000, 12, 8])
end

# ╔═╡ 4ee73e71-b92a-477a-81f5-1e33b56a1608
begin
	mode_t0 = 5.754
	mode_t1 = 5.755
	dt = 2e-6

	frame_0 = convert(Int, round((mode_t0 - 2.0) / dt))
	frame_1 = convert(Int, round((mode_t1 - 2.0) / dt))
end

# ╔═╡ c841e1b7-51ea-409d-b381-726ca3cf2cea
frame_1 - frame_0

# ╔═╡ 5193ed87-e280-41b4-952c-fd4fe83ce1b2
contourf(data_filt[frame_0 + 16,:,:], clims=(-0.075,0.075), 
	color=:bluesreds,
	xlim=(1,8),
	aspect_ratio=1)

# ╔═╡ d91f1d9b-7269-4225-ad94-3853831fe617
begin
	ftime = mode_t0
	anim = @animate for frame ∈ frame_0:frame_1

		title_str = @sprintf "%5d %s t=%8.6fs" shotnr dev ftime
		contourf(data_filt[frame,:,:], clims=(-0.075,0.075), 
			color=:bluesreds,
			aspect_ratio=1,
			title=title_str)
		ftime += dt
	end
	fname = @sprintf "%06d.gif" shotnr
	gif(anim, fname, fps=5)
end

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