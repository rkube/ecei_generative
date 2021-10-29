using HDF5
using Printf

"""
    Loads ECEI frames from hdf5 file
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
