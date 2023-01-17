### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 1804903e-8b95-11ed-01ea-1d6fc7297292
begin
	import Pkg
	Pkg.activate(Base.current_project())
end

# ╔═╡ bfc5ba40-c4bd-433a-ad51-96b0e5dc4d1d
begin
	using HDF5
	using CairoMakie
	using Printf
	using Statistics
	using DSP
	using YAML
end

# ╔═╡ c22ad1fb-0ae9-4741-a9b4-c5a7164fe1be
using ecei_generative

# ╔═╡ 6150b946-d39c-4823-984a-e3b799238577
2+4

# ╔═╡ 6e7ec864-739b-4835-881a-978ecaee0909
pwd()

# ╔═╡ ca5d64e8-d9e2-454a-95d5-b27b0cd6165a
begin
	dt = 2e-6 # Sampling frequency
	t_norm_0 = -0.099 # Start of index used for normalization
	t_norm_1 = -0.089 # End of index used for normalization	
	
	shotnr = 24562
	ds = YAML.load_file("../data/dataset.yaml")
	dev = ds["$(shotnr)"]["dev"]
	t_start = ds["$(shotnr)"]["t_start"]
	t_end = ds["$(shotnr)"]["t_end"]
	mode_t0 = ds["$(shotnr)"]["mode_t0"]
	mode_t1 = ds["$(shotnr)"]["mode_t1"]
	filter_f0 = ds["$(shotnr)"]["filter_f0"]
	filter_f1 = ds["$(shotnr)"]["filter_f1"]
	
	
	datadir = @sprintf "/home/rkube/gpfs/kstar_ecei/%06d/" shotnr
    filename = @sprintf "ECEI.%06d.%s.h5" shotnr dev
	fid = h5open(joinpath(datadir, filename), "r");
end

# ╔═╡ b05e73d0-cf16-4763-a0ee-a155671a5fcc


# ╔═╡ 2343d4c9-2917-4edc-a5f0-9344bed57737
begin
	# Process TriggerTime and build timebase vectors
    TriggerTime = read(HDF5.attributes(fid["/ECEI"])["TriggerTime"]) # Time at trigger
    tbase = (1:5_000_000) .* dt .+ TriggerTime[1] # Time-base used for samples
    tidx_norm = (tbase .> t_norm_0) .& (tbase .< t_norm_1) # Indices that are to be used for normalization
    tidx_all = (tbase .> t_start) .& (tbase .< t_end) # Indices for data that will be returned

	tbase_norm = view(tbase, tidx_norm)
	tbase_sig = view(tbase, tidx_all)
    @show sum(tidx_norm), sum(tidx_all)
end

# ╔═╡ 061a31f2-1a4f-49c7-b48f-e69b6f510f15
begin
    frames_norm = zeros(sum(tidx_norm), 24, 8)
    frames_raw = zeros(sum(tidx_all), 24, 8)
	for ch_v in 1:24
		for ch_h in 1:8
			channel_str = @sprintf "%s%02d%02d" dev ch_v ch_h
			h5var_name = "/ECEI/ECEI_" * channel_str * "/Voltage"
			A = read(fid, h5var_name)
            frames_norm[:, ch_v, ch_h] = A[tidx_norm] * 1e-4
            frames_raw[:, ch_v, ch_h] = A[tidx_all] * 1e-4
		end
	end
end

# ╔═╡ dc749731-1887-4514-a0ef-f28d2df93b8b
contourf(frames_raw[400,:,:])

# ╔═╡ 0e375b82-31f8-4ab1-af6a-a6525f3455d5
tidx_all

# ╔═╡ aae7a1b6-7b8d-4e35-a27c-bacac1720c74
let
	f = Figure()
	a = Axis(f[1, 1])
	lines!(tbase_sig[1:end], frames_raw[1:end, 12, 4])
	lines!(tbase_sig[1:end], frames_raw[1:end, 12, 5])
	f
end

# ╔═╡ 4d491e0d-c604-4860-90d8-ce8c62bcad60
begin
	# Normalize
	offlev = median(frames_norm, dims=1)
	offstd = std(frames_norm, dims=1)

	data_norm = frames_raw.- offlev;

	siglev = median(data_norm, dims=1)
	sigstd = std(data_norm, dims=1)

	data_norm = data_norm ./ mean(data_norm, dims=1) .- 1.0;
	0
end

# ╔═╡ 7567ce2a-3471-49cb-a4e6-273ebf2a9c06
lines(tbase[tidx_all][1:end], data_norm[1:end, 12, 4])

# ╔═╡ b063dc7a-586c-4bd9-b23b-84b0ca7235fc
begin
	ref = 100.0 .* offstd ./ siglev
	ref[siglev .< 0.01] .= 100.0
	bad_channels = ref .> 30.0
	# Mark bottom saturated signals
	bad_channels[offstd .< 1e-3] .= true
	# Mark top saturated signals
	bad_channels[sigstd .< 1e-3] .= true
	bad_channels = dropdims(bad_channels, dims=1)
end

# ╔═╡ 3b5a9394-403b-4b5e-abb4-4b19333a2f27
ipol_dict = generate_ip_index_set(bad_channels)

# ╔═╡ d6906710-1636-4daf-976d-adb1f77af559
begin
	data_norm_ip = similar(data_norm)
	# ipol_dict = generate_ip_index_set(bad_channels)
	Threads.@threads for i in 1:size(data_norm)[1]
		data_norm_ip[i,:,:] = ip_bad_values(data_norm[i,:,:], ipol_dict)
	end
end

# ╔═╡ e984a432-1163-4223-bfea-1fa31c016677
begin
	responsetype = Bandpass(filter_f0, filter_f1; fs=1.0/dt) 
	designmethod = Butterworth(4)
	my_filter = digitalfilter(responsetype, designmethod)
end

# ╔═╡ 944d01d6-904b-4d24-8539-68e6e458c3be
begin
	data_norm_filt = similar(data_norm)
	Threads.@threads for ch_v in 1:size(data_norm_filt)[2]
		for ch_h in 1:size(data_norm_filt)[3]
			data_norm_filt[:, ch_v, ch_h] = filt(my_filter, data_norm_ip[:, ch_v, ch_h])
		end
	end
end

# ╔═╡ 67297ec4-b31e-482e-b849-ef1f78e0f609
let
	f = Figure()
	a = Axis(f[1, 1])
	lines!(tbase_sig, data_norm_ip[:, 12, 6])
	lines!(tbase_sig, data_norm_filt[:, 12, 6])
	f
end


# ╔═╡ 030d8452-c1e1-4aec-9607-972d0ab660c4
t_start, mode_t0, mode_t1, t_end

# ╔═╡ 4db4a3f7-1a61-421f-bd77-8d393368aadb
begin
	# mode_t0 = t_start
	# mode_t1 = t_start + 1e-3

	#frame_0 = Int(ceil((mode_t0 - t_start + eps(typeof(t_start))) / dt))
	#frame_1 = Int(ceil((mode_t1 - t_start) / dt)) - 10
	frame_0 = Int(ceil((mode_t0 - t_start + eps(typeof(t_start))) / dt))
	frame_1 = Int(ceil((mode_t1 - t_start) / dt)) - 10

	frame_0, frame_1
end

# ╔═╡ 1ce0626f-a214-4515-9808-302b542987c8
size(data_norm_filt)

# ╔═╡ e5adc0f8-bbfa-4c5c-8c5c-d052c86375d4
data_norm_filt[1, 1, end:-1:1]

# ╔═╡ 8728bc45-f741-4826-8beb-4a87ae55da78
data_norm_filt[1, 1, :]

# ╔═╡ 79f4ec19-518a-4791-a3af-02db5efae9b0
frame_0

# ╔═╡ defed8b0-ad7b-47de-ac23-658c3a4b8a29
frame_0, frame_1

# ╔═╡ 22772c8b-886d-4853-a3e5-9419bbdc53bb
lines(mode_t0 .+ tbase[frame_0:frame_1], data_norm_filt[frame_0:frame_1, 12, 6])

# ╔═╡ e0cf02e9-a5d6-44c0-ac18-96f842b618dc
let
	cf_levels = LinRange(-0.15, 0.15, 16)
	ix_t = Observable(frame_0)

	frame_data = @lift data_norm_filt[$ix_t, :, end:-1:1]'
	
	fig, ax, cf = contourf(frame_data,
		                   levels=cf_levels,
		       			   colormap=:vik,
		                   axis=(title = @lift("t= $(tbase_sig[$ix_t])s"), aspect=1/3))
	Colorbar(fig[1, 2], cf)

	record(fig, "$(shotnr)_reorder.gif", frame_0:frame_1; framerate=10) do t
 		ix_t[] = t
	end

	fig
end

# ╔═╡ c06111fb-afd1-4537-92c0-00abb29d291e
# let
# 	ftime = mode_t0
# 	anim = @animate for frame ∈ frame_0:frame_1

# 		title_str = @sprintf "%5d %s t=%8.6fs" shotnr dev ftime
# 		heatmap(data_norm_filt[frame, :,:], clims=(-0.15,0.15), 
# 			levels=16,
# 			color=:vik,
# 			xlim=(1, 8),
# 			aspect_ratio=1,
# 			title=title_str)
# 		ftime += dt
# 	end
# 	fname_anim = @sprintf "%06d_reorder.gif" shotnr
# 	gif(anim, fname_anim, fps=5)
# end

# ╔═╡ Cell order:
# ╠═1804903e-8b95-11ed-01ea-1d6fc7297292
# ╠═6150b946-d39c-4823-984a-e3b799238577
# ╠═bfc5ba40-c4bd-433a-ad51-96b0e5dc4d1d
# ╠═c22ad1fb-0ae9-4741-a9b4-c5a7164fe1be
# ╠═6e7ec864-739b-4835-881a-978ecaee0909
# ╠═ca5d64e8-d9e2-454a-95d5-b27b0cd6165a
# ╠═b05e73d0-cf16-4763-a0ee-a155671a5fcc
# ╠═2343d4c9-2917-4edc-a5f0-9344bed57737
# ╠═061a31f2-1a4f-49c7-b48f-e69b6f510f15
# ╠═dc749731-1887-4514-a0ef-f28d2df93b8b
# ╠═0e375b82-31f8-4ab1-af6a-a6525f3455d5
# ╠═aae7a1b6-7b8d-4e35-a27c-bacac1720c74
# ╠═4d491e0d-c604-4860-90d8-ce8c62bcad60
# ╠═7567ce2a-3471-49cb-a4e6-273ebf2a9c06
# ╠═b063dc7a-586c-4bd9-b23b-84b0ca7235fc
# ╠═3b5a9394-403b-4b5e-abb4-4b19333a2f27
# ╠═d6906710-1636-4daf-976d-adb1f77af559
# ╠═e984a432-1163-4223-bfea-1fa31c016677
# ╠═944d01d6-904b-4d24-8539-68e6e458c3be
# ╠═67297ec4-b31e-482e-b849-ef1f78e0f609
# ╠═030d8452-c1e1-4aec-9607-972d0ab660c4
# ╠═4db4a3f7-1a61-421f-bd77-8d393368aadb
# ╠═1ce0626f-a214-4515-9808-302b542987c8
# ╠═e5adc0f8-bbfa-4c5c-8c5c-d052c86375d4
# ╠═8728bc45-f741-4826-8beb-4a87ae55da78
# ╠═79f4ec19-518a-4791-a3af-02db5efae9b0
# ╠═defed8b0-ad7b-47de-ac23-658c3a4b8a29
# ╠═22772c8b-886d-4853-a3e5-9419bbdc53bb
# ╠═e0cf02e9-a5d6-44c0-ac18-96f842b618dc
# ╠═c06111fb-afd1-4537-92c0-00abb29d291e
