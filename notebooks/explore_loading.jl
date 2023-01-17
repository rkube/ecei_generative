### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ d50686b0-83ec-4e35-b9c7-748ae2278bc5
begin
	using Pkg
	Pkg.activate(Base.current_project())
end

# ╔═╡ 71359fe0-5dd5-11ec-25e8-1715b363af65
begin
	using Plots
	using HDF5
	using Printf
	using Statistics
	using StatsBase
	using Interpolations
	using DSP
end

# ╔═╡ f440bdf6-1019-4c1b-8b10-d0cbb0c15411
begin
	shotnr = 25878
	dev = "GR"
	t_start = 5.0
	t_end = 7.0
	dt = 2e-6 # Sampling frequency
	t_norm_0 = -0.099 # Start of index used for normalization
	t_norm_1 = -0.089 # End of index used for normalization	
	datadir = @sprintf "/home/rkube/gpfs/KSTAR/%06d/" shotnr
    filename = @sprintf "ECEI.%06d.%s.h5" shotnr dev
	fid = h5open(joinpath(datadir, filename), "r")
end

# ╔═╡ 27920740-b434-47b3-8a8c-8a36332f4a6b
begin
	# Process TriggerTime and build timebase vectors
    TriggerTime = read(attributes(fid["/ECEI"])["TriggerTime"]) # Time at trigger
    tbase = (1:5_000_000) .* dt .+ TriggerTime[1] # Time-base used for samples
    tidx_norm = (tbase .> t_norm_0) .& (tbase .< t_norm_1) # Indices that are to be used for normalization
    tidx_all = (tbase .> t_start) .& (tbase .< t_end) # Indices for data that will be returned
    @show sum(tidx_norm), sum(tidx_all)
end

# ╔═╡ 0f90523a-8c0e-4fa1-8c7b-f32f29f04151
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

# ╔═╡ b3b4fb21-71dc-4cc1-a50a-3887b19b1a25
contourf(frames_raw[4000,:,:])

# ╔═╡ cf370d7e-1804-4e08-8e03-4f0e272cd1e4
begin
	plot(tbase[tidx_all][1:100:end], frames_raw[1:100:end, 12, 4])
	plot!(tbase[tidx_all][1:100:end], frames_raw[1:100:end, 12, 5])
end

# ╔═╡ 0ad2fc36-9760-4dd6-8dbd-706b83f89427
begin
	# Normalize
	offlev = median(frames_norm, dims=1)
	offstd = std(frames_norm, dims=1)

	data_norm = frames_raw.- offlev;

	siglev = median(data_norm, dims=1)
	sigstd = std(data_norm, dims=1)

	data_norm = data_norm ./ mean(data_norm, dims=1) .- 1.0
end

# ╔═╡ 754bb2a0-c80a-4da6-9473-9d56c932499b
plot(tbase[tidx_all][1:100:end], data_norm[1:100:end, 12, 4])

# ╔═╡ b53ef757-542a-4601-a8c2-cce0a462f2e6
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

# ╔═╡ 283ed241-e2f9-41ab-aec8-21298f50fcd5
function generate_ip_index_set(bad_channels)
	# This function calculates the neighbors for pixels marked as true in
	# a 2d boolean array.
	# Input: bad_channels, BitArray{2}.
	# Output: Dict(bad_channel -> [list of neighboring good pixels])
	R = CartesianIndices(bad_channels) # Create cartesian indices
	
	Ifirst, Ilast = first(R), last(R) # Range for interpolation
	I_one = oneunit(Ifirst)           # Unit square around a pixel

	ipol_dict = Dict()
	
	# Iterate over all bad pixels and create a list of good neighbor channels 
	# for this pixel.
	for I in findall(bad_channels)
		@show I
		ipol_list_ch = []
		for J in max(Ifirst, I - I_one):min(Ilast, I + I_one)
			if bad_channels[J] == false
				push!(ipol_list_ch, J)
			end
		end
		@show ipol_list_ch
		merge!(ipol_dict, Dict(I => ipol_list_ch))
	end
	return ipol_dict
end

# ╔═╡ 21e29a9f-e6bf-4b35-9fac-bf91fe1e9e5f
function ip_bad_values(field, ipol_dict)
	R = CartesianIndices(field) # Create cartesian indices
	field_ip = similar(field)   # Field with interpolated values
	field_ip[:] = field[:]      # Copy all values
	
	Ifirst, Ilast = first(R), last(R) # Range for interpolation
	I_one = oneunit(Ifirst) # Unit square around a pixel

	for bad_px in keys(ipol_dict)
		bad_px_entries = ipol_dict[bad_px]

		# We can't interpolate pixels that have no valid pixels as neighbors, so we skip them 
		if length(bad_px_entries) == 0
			field_ip[bad_px] = zero(eltype(field))
			continue 
		end
		
		# Interpolate the bad pixel using the average of neighboring pixels
		ip_val = 0.0
		for J in bad_px_entries
			ip_val += field[J]
		end
		ip_val = ip_val / length(ipol_dict[bad_px])
		field_ip[bad_px] = ip_val
	end
	field_ip
end

# ╔═╡ 51f76fb5-72d6-404f-a736-6e843f57adc3
ipol_dict = generate_ip_index_set(bad_channels)

# ╔═╡ e5af61f2-0e70-44aa-816c-f382a41562c4
for k in keys(ipol_dict)
	@show k
end

# ╔═╡ 2279bdfe-2903-4abf-8bb4-bad5dae0f7a3
contourf(ip_bad_values(data_norm[1,:,:], ipol_dict))

# ╔═╡ 3b7d5b38-7071-4abb-b44a-f210b53c772a
begin
	data_norm_ip = similar(data_norm)
	# ipol_dict = generate_ip_index_set(bad_channels)
	Threads.@threads for i in 1:size(data_norm)[1]
		data_norm_ip[i,:,:] = ip_bad_values(data_norm[i,:,:], ipol_dict)
	end
end

# ╔═╡ 63f533ae-7165-40cd-9c78-ee69cd62b50c
size(data_norm)

# ╔═╡ d6667ab4-cb0d-40d2-b68b-b9663d8af927
begin
	responsetype = Bandpass(35000, 50000; fs=1.0/dt) 
	designmethod = Butterworth(4)
	my_filter = digitalfilter(responsetype, designmethod)
end

# ╔═╡ 004a1b8a-ffe5-47bb-960a-bfe894cd5ea7
size(data_norm_ip)

# ╔═╡ 881372e2-1c35-481f-9a49-f24f10ff75e8
begin
	data_norm_filt = similar(data_norm)
	Threads.@threads for ch_v in 1:size(data_norm_filt)[2]
		for ch_h in 1:size(data_norm_filt)[3]
			data_norm_filt[:, ch_v, ch_h] = filt(my_filter, data_norm_ip[:, ch_v, ch_h])
		end
	end
end

# ╔═╡ b794e05b-302d-4c1f-9d9d-254aa2147a7f
begin
	plot(data_norm_ip[1:1000, 12, 6])
	plot!(data_norm_filt[1:1000, 12, 6])
end

# ╔═╡ 26877ccf-477d-44eb-b12f-823a69f7b94a
contourf(data_norm_filt[450015,:,:], clims=(-0.1,0.1), color=:bluesreds)

# ╔═╡ 489e7d1f-a711-4ec8-9672-beaf500f5beb
# histogram(data_norm_filt[:])

# ╔═╡ b8ce6b5d-bc3c-49a5-9047-c5d58fd2b3ca
tbase[tidx_all][450000]

# ╔═╡ Cell order:
# ╠═d50686b0-83ec-4e35-b9c7-748ae2278bc5
# ╠═71359fe0-5dd5-11ec-25e8-1715b363af65
# ╠═f440bdf6-1019-4c1b-8b10-d0cbb0c15411
# ╠═27920740-b434-47b3-8a8c-8a36332f4a6b
# ╠═0f90523a-8c0e-4fa1-8c7b-f32f29f04151
# ╠═b3b4fb21-71dc-4cc1-a50a-3887b19b1a25
# ╠═cf370d7e-1804-4e08-8e03-4f0e272cd1e4
# ╠═0ad2fc36-9760-4dd6-8dbd-706b83f89427
# ╠═754bb2a0-c80a-4da6-9473-9d56c932499b
# ╠═b53ef757-542a-4601-a8c2-cce0a462f2e6
# ╠═283ed241-e2f9-41ab-aec8-21298f50fcd5
# ╠═e5af61f2-0e70-44aa-816c-f382a41562c4
# ╠═21e29a9f-e6bf-4b35-9fac-bf91fe1e9e5f
# ╠═2279bdfe-2903-4abf-8bb4-bad5dae0f7a3
# ╠═51f76fb5-72d6-404f-a736-6e843f57adc3
# ╠═3b7d5b38-7071-4abb-b44a-f210b53c772a
# ╠═63f533ae-7165-40cd-9c78-ee69cd62b50c
# ╠═d6667ab4-cb0d-40d2-b68b-b9663d8af927
# ╠═004a1b8a-ffe5-47bb-960a-bfe894cd5ea7
# ╠═881372e2-1c35-481f-9a49-f24f10ff75e8
# ╠═b794e05b-302d-4c1f-9d9d-254aa2147a7f
# ╠═26877ccf-477d-44eb-b12f-823a69f7b94a
# ╠═489e7d1f-a711-4ec8-9672-beaf500f5beb
# ╠═b8ce6b5d-bc3c-49a5-9047-c5d58fd2b3ca
