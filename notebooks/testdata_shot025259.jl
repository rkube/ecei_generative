### A Pluto.jl notebook ###
# v0.18.4

using Markdown
using InteractiveUtils

# ╔═╡ f533c82f-a09c-47ec-8387-50ce6efbc943
begin
	import Pkg
	Pkg.activate("/home/rkube/repos/ecei_generative")
	Pkg.instantiate()
	Pkg.add("Plots")

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

# ╔═╡ a0f07049-0a6c-440a-a7e7-b74771371af2
findall(x -> x == true, Bool[0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0; 0 0 1 1 0 0 0 0; 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0; 1 1 1 0 0 0 0 0; 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0; 0 0 0 0 0 1 0 0])

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

# ╔═╡ a4566bf6-a910-4210-adb0-4eae59e74ee7


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

# ╔═╡ 355d51f5-23e4-4fb6-aeb2-006302bee38f
rg = maximum(data_filt) - minimum(data_filt)

# ╔═╡ a4a66afc-95c8-42a2-ade7-555079655761
begin
	bad_idx = findall(arr -> abs.(arr) .> 2.0, data_filt)
	bad_idx_2 = [(i[1], i[2]) for i in bad_idx]
end

# ╔═╡ 835d7f81-fbff-42a5-a466-08198a091bbf
bad_idx[1]

# ╔═╡ 13b774d6-3240-441e-ad82-bbb89127ab38
length(bad_idx)

# ╔═╡ 7e45f3d4-1d29-4728-baf5-021f53ad4a54
contourf(data_filt[:, :, 87403])

# ╔═╡ 27b2ac07-3f86-4187-96c2-68af53cfdd44
data_filt[20, 8, 87522] = 0f0

# ╔═╡ 4c8edadc-481d-48ab-a9e5-52637761472f
histogram(data_filt[:])

# ╔═╡ 8cfa9ca7-b3fb-4fc9-ab57-81cb2dcebedb
begin
	data_norm = 2f0 * (data_filt .- minimum(data_filt)) / (maximum(data_filt) - minimum(data_filt)) .- 1f0;
	data_std = (data_filt .- mean(data_filt)) ./ std(data_filt);
	0
end

# ╔═╡ 9dec3253-633c-4d86-bff9-cf429ca67dcc
histogram(data_norm[:])

# ╔═╡ 77e41325-57b3-4703-b0ec-8a3ba473b601
histogram(data_std[:])

# ╔═╡ 21362555-39a2-4778-954d-dbc8f1da1c5b
maximum(data_filt), minimum(data_filt), mean(data_filt), std(data_filt)

# ╔═╡ 3a8d30e6-6a10-476b-abd8-8bf0b63591a5
maximum(data_norm), minimum(data_norm), mean(data_norm), std(data_norm)

# ╔═╡ 4533123d-43a8-489b-919d-9414e1979d77
maximum(data_std), minimum(data_std), mean(data_std), std(data_std)

# ╔═╡ eaad041c-bcfd-412c-9f84-7d1e4555304a


# ╔═╡ Cell order:
# ╠═f533c82f-a09c-47ec-8387-50ce6efbc943
# ╠═7d8169b8-5838-41d5-be00-c508676df9e2
# ╠═2dc5b778-579c-47dd-833a-195d2203d4f1
# ╠═3040e530-2c83-4e19-afa1-a7923e725f9f
# ╠═a0f07049-0a6c-440a-a7e7-b74771371af2
# ╠═ac899c28-a8b5-4122-a093-b1f54589cfce
# ╠═4ee73e71-b92a-477a-81f5-1e33b56a1608
# ╠═c841e1b7-51ea-409d-b381-726ca3cf2cea
# ╠═5193ed87-e280-41b4-952c-fd4fe83ce1b2
# ╠═a4566bf6-a910-4210-adb0-4eae59e74ee7
# ╠═d91f1d9b-7269-4225-ad94-3853831fe617
# ╠═355d51f5-23e4-4fb6-aeb2-006302bee38f
# ╠═a4a66afc-95c8-42a2-ade7-555079655761
# ╠═835d7f81-fbff-42a5-a466-08198a091bbf
# ╠═13b774d6-3240-441e-ad82-bbb89127ab38
# ╠═7e45f3d4-1d29-4728-baf5-021f53ad4a54
# ╠═27b2ac07-3f86-4187-96c2-68af53cfdd44
# ╠═4c8edadc-481d-48ab-a9e5-52637761472f
# ╠═9dec3253-633c-4d86-bff9-cf429ca67dcc
# ╠═77e41325-57b3-4703-b0ec-8a3ba473b601
# ╠═8cfa9ca7-b3fb-4fc9-ab57-81cb2dcebedb
# ╠═21362555-39a2-4778-954d-dbc8f1da1c5b
# ╠═3a8d30e6-6a10-476b-abd8-8bf0b63591a5
# ╠═4533123d-43a8-489b-919d-9414e1979d77
# ╠═eaad041c-bcfd-412c-9f84-7d1e4555304a
