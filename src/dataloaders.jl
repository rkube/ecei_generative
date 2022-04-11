using HDF5
using Printf
using Statistics
using DSP

export generate_ip_index_set, ip_bad_values, get_framedata, load_from_hdf

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


@doc """

    load_from_hdf

Load ECEi frames from HDF5 - raw data with local pre-processing

Parameters:
* t_start - Start time for returned ECEI data. In seconds.
* t_end - End time for returned ECEI data. In seconds.
* f_filt_lo - Lower end of bandpass filter. In Hz.
* f_filt_hi - Upper end of bandpass filter. In Hz.
* datadir - Directory where HDF5 files are locaed
* shotnr - Shot number
* dev - ECEI device
Keywords: 
* t_norm_0 - Start point of interval used for normalization. Default=-0.099. In seconds.
* t_norm_1 - End point of interval used for normalization. Default=-0.089. In seconds.
* dt - Sampling Time. Default = 2e-6. In seconds.
"""
function load_from_hdf(t_start, t_end, f_filt_lo, f_filt_hi, datadir, shotnr, dev; t_norm_0=-0.099, t_norm_1=-0.089, dt=2e-6)
    # Construct the filename
    filename = @sprintf "ECEI.%06d.%s.h5" shotnr dev

    # Open HDF5 file
    fid = h5open(joinpath(datadir, filename), "r")
    
    # Process TriggerTime and build timebase vectors
    TriggerTime = read(attributes(fid["/ECEI"])["TriggerTime"]) # Time at trigger
    tbase = (1:5_000_000) .* dt .+ TriggerTime[1] # Time-base used for samples
    tidx_norm = (tbase .> t_norm_0) .& (tbase .< t_norm_1) # Indices that are to be used for normalization
    tidx_all = (tbase .> t_start) .& (tbase .< t_end) # Indices for data that will be returned

    # Allocate memory for arrays that will hold data used for normalization and the actual data
    # Allocate such that individual channel time series lie consecutive in memory since we later
    # need to access them for normalization and filtering etc.
    frames_norm = zeros(sum(tidx_norm), 24, 8)
    frames_raw = zeros(sum(tidx_all), 24, 8)

    # Read data from HDF5 file
    for ch_v in 1:24 # iterate over vertical channels
        for ch_h in 1:8 # iterate over horizontal channels
            channel_str = @sprintf "%s%02d%02d" dev ch_v ch_h
		    h5_var_name = "/ECEI/ECEI_" * channel_str * "/Voltage"
		    A = read(fid, h5_var_name)
            frames_norm[:, ch_v, ch_h] = A[tidx_norm] * 1e-4
            frames_raw[:, ch_v, ch_h] = A[tidx_all] * 1e-4
	    end
	end
    println("Read data from hdf5")

    # Calculate offsets, normalize, etc.
	# Normalize
	offlev = median(frames_norm, dims=1)
	offstd = std(frames_norm, dims=1)

	data_norm = frames_raw .- offlev;

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
    data_norm_ip = zeros(Float32, size(data_norm))
	Threads.@threads for i in 1:size(data_norm)[1]
		data_norm_ip[i, :, :] = ip_bad_values(data_norm[i, :, :], ipol_dict)
	end

    # Apply bandpass filter.
    # TODO: Remove hard-coded pass and stop band
	responsetype = Bandpass(f_filt_lo, f_filt_hi; fs=1.0/dt) 
	designmethod = Butterworth(4)
	my_filter = digitalfilter(responsetype, designmethod)

    data_norm_filt = zeros(Float32, size(data_norm))
	Threads.@threads for ch_v in 1:size(data_norm_filt)[2]
		for ch_h in 1:size(data_norm_filt)[3]
			data_norm_filt[:, ch_v, ch_h] = filt(my_filter, data_norm_ip[:, ch_v, ch_h])
		end
	end

    # Return the normalized, frequency-filtered data
    return permutedims(data_norm_filt, [2, 3, 1])
end
