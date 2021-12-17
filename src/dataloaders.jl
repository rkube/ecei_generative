using HDF5
using Printf
using Statistics
using DSP

"""
   Returns a list of indices that are to be used for interpolation around bad channels.
   bad_channels: 2d bit-mask where a bad channel is marked as true
   returns:
        ipol_dict: dictionary. Key: bad channel. Value: List of indices to be used for interpolation at that pixel.
"""
function generate_ip_index_set(bad_channels)
	R = CartesianIndices(bad_channels) # Create cartesian indices
	
	Ifirst, Ilast = first(R), last(R) # Range for interpolation
	I_one = oneunit(Ifirst) # Unit square around a pixel

	ipol_dict = Dict()
	
	for I in findall(bad_channels)
		ipol_list_ch = []
		for J in max(Ifirst, I - I_one):min(Ilast, I + I_one)
			if bad_channels[J] == false
				push!(ipol_list_ch, J)
			end
		end
		merge!(ipol_dict, Dict(I => ipol_list_ch))
	end
	return ipol_dict
end


"""
    Interpolates the bad pixels in a 2d-array using the pixels specified in ipol_dict.
"""

function ip_bad_values(field, ipol_dict)
	#field = data_norm[1,:,:]
	R = CartesianIndices(field) # Create cartesian indices
	field_ip = similar(field) # Field with interpolated values
	field_ip[:] = field[:] # Copy all values

	Ifirst, Ilast = first(R), last(R) # Range for interpolation
	I_one = oneunit(Ifirst) # Unit square around a pixel

    for bad_px in keys(ipol_dict)
		bad_px_entries = ipol_dict[bad_px]

		ip_val = 0.0
		for J in bad_px_entries
			ip_val += field[J]
		end
		ip_val = ip_val / length(ipol_dict[bad_px])
		field_ip[bad_px] = ip_val
	end
	field_ip
end

"""
    Loads ECEI frames from hdf5 file - pre-processed on Cori using DELTA
"""
function get_framedata(shotnr=25259, dev="GT", chunk=137, datadir="/home/rkube/repos/ecei_generative/data")
   # Construct filename and dataset name
   # Files are written on cori, see ~/delta_misc/
   fname = @sprintf "frames_%05d_%s_%03d.h5" shotnr dev chunk
   dset_name = @sprintf "frame_%03d" chunk
   # Open hdf5 file and read dataset
   fid = h5open(joinpath(datadir, fname))
   # Permute dimensions: dim1: rows, dim2: columns, dim3: time
   frame_data = permutedims(fid[dset_name][:,:,:], [3, 2, 1])
   close(fid)
   convert(Array{Float32, 3}, frame_data)
end


"""
    Load ECEi frames from HDF5 - raw data with local pre-processing
"""
function load_from_hdf(t_start, t_end, datadir, shotnr, dev)


    # Array that holds the raw data
    raw_frames = zeros(5_000_000, 24, 8)
    # Construct the filename
    filename = @sprintf "ECEI.%06d.%s.h5" shotnr dev

    # Open HDF5 file
    fid = h5open(joinpath(datadir, filename), "r")

	TriggerTime = read(attributes(fid["/ECEI"])["TriggerTime"]) # Time at trigger
    # Read data from HDF5 file
    for ch_v in 1:24 # iterate over vertical channels
        for ch_h in 1:8 # iterate over horizontal channels
            channel_str = @sprintf "%s%02d%02d" dev ch_v ch_h
			#ch_idx = ch_to_idx(channel_str)
			h5_var_name = "/ECEI/ECEI_" * channel_str * "/Voltage"
			A = read(fid, h5_var_name)
			raw_frames[:, ch_v, ch_h] = A[:] .* 1e-4
		end
	end
	dt = 2e-6 # Sampling frequency
	t_norm_0 = -0.099 # Start of index used for normalization
	t_norm_1 = -0.089 # End of index used for normalization
	tbase = (1:5_000_000) .* dt .+ TriggerTime[1] # Time-base used for samples
	tidx_norm = (tbase .> t_norm_0) .& (tbase .< t_norm_1) # Indices that are to be 

    # Calculate offsets, normalize, etc.
	# Normalize
	offlev = median(raw_frames[tidx_norm, :, :], dims=1)
	offstd = std(raw_frames[tidx_norm, :, :], dims=1)


	data_norm = raw_frames[1_000_000:end, :, :] .- offlev;

	siglev = median(data_norm, dims=1)
	sigstd = std(data_norm, dims=1)

	data_norm = data_norm ./ mean(data_norm, dims=1) .- 1.0

    # Mark bad channels
	ref = 100.0 .* offstd ./ siglev
	ref[siglev .< 0.01] .= 100.0
	bad_channels = ref .> 30.0
	# Mark bottom saturated signals
	bad_channels[offstd .< 1e-3] .= true
	# Mark top saturated signals
	bad_channels[sigstd .< 1e-3] .= true
	bad_channels = dropdims(bad_channels, dims=1)

    # Calculate the pixels that are to be used for interpolation of bad pixels
    ipol_dict = generate_ip_index_set(bad_channels)
    # Interpolate bad pixels. This should be done multi-threaded
	data_norm_ip = similar(data_norm)
	Threads.@threads for i in 1:size(data_norm)[1]
		data_norm_ip[i,:,:] = ip_bad_values(data_norm[i,:,:], ipol_dict)
	end

    # Apply bandpass filter.
    # TODO: Remove hard-coded pass and stop band
	responsetype = Bandpass(35000, 50000; fs=1.0/dt) 
	designmethod = Butterworth(4)
	my_filter = digitalfilter(responsetype, designmethod)

	data_norm_filt = similar(data_norm)
	Threads.@threads for ch_v in 1:size(data_norm_filt)[2]
		for ch_h in 1:size(data_norm_filt)[3]
			data_norm_filt[:, ch_v, ch_h] = filt(my_filter, data_norm_ip[:, ch_v, ch_h])
		end
	end

    # Return the normalized, frequency-filtered data
    return data_norm_filt
end
