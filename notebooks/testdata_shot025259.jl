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
	using Plots
	using Printf
	using Statistics
	using StatsBase
end

# ╔═╡ 7d8169b8-5838-41d5-be00-c508676df9e2


# ╔═╡ 2dc5b778-579c-47dd-833a-195d2203d4f1
begin
	t_start = 2.0
	t_end = 3.0
	filter_f0 = 30000
	filter_f1 = 50000
	shotnr = 25259
	dev = "GT"
	datadir = @sprintf "/home/rkube/gpfs/KSTAR/%06d" shotnr
end

# ╔═╡ 3040e530-2c83-4e19-afa1-a7923e725f9f
data_filt = load_from_hdf(t_start, t_end, filter_f0, filter_f1, datadir, shotnr, dev);

# ╔═╡ ac899c28-a8b5-4122-a093-b1f54589cfce
begin
	plot(data_filt[12, 6, 370_000:380_000])
	plot!(data_filt[12, 7, 370_000:380_000])
	plot!(data_filt[12, 8, 370_000:380_000])
end

# ╔═╡ 4ee73e71-b92a-477a-81f5-1e33b56a1608
begin
	mode_t0 = 2.664
	mode_t1 = 2.665
	dt = 2e-6

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

# ╔═╡ eaad041c-bcfd-412c-9f84-7d1e4555304a
begin
	# Plot data normalization and look at clipping
	data_tr = clamp.(data_filt, -0.15, 0.15);
	tr = fit(UnitRangeTransform, data_tr[:]);
	data_unif = StatsBase.transform(tr, data_tr[:]);

	tr = fit(ZScoreTransform, data_tr[:]);
	data_std = StatsBase.transform(tr, data_tr[:]);
end

# ╔═╡ 18c0caf8-63fa-49eb-9231-b8a80c25e81f
sum(abs.(data_filt) .< 1e-3) / length(data_filt)

# ╔═╡ 8cc3516a-c7d1-4b6a-a0d8-f506c60f12c5
findall(x -> abs(x) < 1e-3, data_filt[:, :, frame_0])

# ╔═╡ 2321746c-ca34-4f44-af8c-ab4b07e1d974
begin
	frame = frame_0 + 10
	p = contourf(data_filt[:, :, frame],  clims=(-0.075,0.075))
	idx = findall(x -> abs(x) < 1e-3, data_filt[:, :, frame])
	plot!(p, [i[2] for i in idx], [i[1] for i in idx], seriestype=:scatter, color=:black, ms=8)
	p
end

# ╔═╡ 4cfe783d-9256-4a08-933f-5b20cc197d1c


# ╔═╡ 93a63569-ae97-4059-859e-94db0c575737
histogram(data_filt[:])

# ╔═╡ fd07241d-da8e-4a86-bd4e-643fffaf3ac3
histogram(data_norm[:])

# ╔═╡ 09361635-e427-45d3-a8b8-619ad102e405
histogram(data_std[:])

# ╔═╡ Cell order:
# ╠═f533c82f-a09c-47ec-8387-50ce6efbc943
# ╠═7d8169b8-5838-41d5-be00-c508676df9e2
# ╠═2dc5b778-579c-47dd-833a-195d2203d4f1
# ╠═3040e530-2c83-4e19-afa1-a7923e725f9f
# ╠═ac899c28-a8b5-4122-a093-b1f54589cfce
# ╠═4ee73e71-b92a-477a-81f5-1e33b56a1608
# ╠═c841e1b7-51ea-409d-b381-726ca3cf2cea
# ╠═5193ed87-e280-41b4-952c-fd4fe83ce1b2
# ╠═d91f1d9b-7269-4225-ad94-3853831fe617
# ╠═eaad041c-bcfd-412c-9f84-7d1e4555304a
# ╠═18c0caf8-63fa-49eb-9231-b8a80c25e81f
# ╠═8cc3516a-c7d1-4b6a-a0d8-f506c60f12c5
# ╠═2321746c-ca34-4f44-af8c-ab4b07e1d974
# ╠═4cfe783d-9256-4a08-933f-5b20cc197d1c
# ╠═93a63569-ae97-4059-859e-94db0c575737
# ╠═fd07241d-da8e-4a86-bd4e-643fffaf3ac3
# ╠═09361635-e427-45d3-a8b8-619ad102e405
