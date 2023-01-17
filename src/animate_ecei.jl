#using HDF5
using CairoMakie
#using Printf
#using Statistics
#using DSP
using YAML

using ecei_generative

#using ecei_generative

"""
Animate the interesting mode times in the dataset file.
"""

dt = 2e-6 # Sampling frequency
t_norm_0 = -0.099 # Start of index used for normalization
t_norm_1 = -0.089 # End of index used for normalization	

shotnr = 25522
ds = YAML.load_file("data/dataset.yaml")

dev = ds["$(shotnr)"]["dev"]
t_start = ds["$(shotnr)"]["t_start"]
t_end = ds["$(shotnr)"]["t_end"]
mode_t0 = ds["$(shotnr)"]["mode_t0"]
mode_t1 = ds["$(shotnr)"]["mode_t1"]
filter_f0 = ds["$(shotnr)"]["filter_f0"]
filter_f1 = ds["$(shotnr)"]["filter_f1"]


datadir = "/home/rkube/gpfs/kstar_ecei/" * lpad(shotnr, 6, "0")
data_norm, tbase_norm = load_from_hdf(t_start, t_end, filter_f0, filter_f1, datadir, shotnr, dev)


frame_0 = Int(ceil((mode_t0 - t_start + eps(typeof(t_start))) / dt))
frame_1 = Int(ceil((mode_t1 - t_start) / dt)) - 10

cf_levels = LinRange(-0.15, 0.15, 16)
ix_t = Observable(frame_0)

frame_data = @lift data_norm[:, end:-1:1, $ix_t]'

fig, ax, cf = contourf(frame_data,
                       levels=cf_levels,
                          colormap=:vik,
                       axis=(title = @lift("t= $(tbase_norm[$ix_t])s"), aspect=1/3))
Colorbar(fig[1, 2], cf)

record(fig, "$(shotnr)_reorder.gif", frame_0:frame_1; framerate=10) do t
     ix_t[] = t
end

fig