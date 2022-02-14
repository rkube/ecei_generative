using ecei_generative
using Plots
using Printf
using DataFrames
using Clustering
using MultivariateStats
using Flux
using EvalMetrics

import MultivariateStats

# Monkey patching. See https://github.com/JuliaStats/MultivariateStats.jl/pull/167

import StatsBase.predict
predict(M::PCA{T}, x::AbstractVecOrMat{T}) where     {T<:Real} = transpose(M.proj) * centralize(x, M.mean)

# Create test data structure
df = DataFrame(shotnr=Int[], dev=String[], tstart=Float64[], tend=Float64[], f0=Int[], f1=[])
push!(df, [25259, "GT", 2.659, 2.660, 35000, 50000]) # 3/2 island
push!(df, [25260, "GT", 5.909, 5.910, 35000, 50000]) # 3/2 island
# # Dataset  for 25878 has nans14G
# # push!(df, [25878, "GR", 3.90, 3.901, 35000, 5000    0])
push!(df, [25879, "GR", 5.904, 5.905, 5000, 20000]) # 2/1 island
push!(df, [25880, "GR", 5.404, 5.405, 1000, 20000]) # 2/1 island
push!(df, [25897, "GT", 5.914, 5.915, 1000, 20000]) # 2/1 island
#   #push!(df,   [25973, "GR", 5.754, 5.755, 1000, 20000]) # 2/1 + 3/2
push!(df, [25978, "GR", 5.754, 5.755, 20000, 40000]) # 3/2 island

data_list = []
for row in eachrow(df)
    datadir = @sprintf "/home/rkube/gpfs/KSTAR/%06d" row.shotnr
    data_filt = load_from_hdf(row.tstart, row.tend, row.f0, row.f1, datadir, row.shotnr, row.dev);
    push!(data_list, data_filt)
end 

# Create ground truth labels. Each dataset has 499 samples in it.
labels = vcat(ones(499), ones(499), 2 * ones(499), 2 * ones(499), 2 * ones(499), ones(499));

# Concatenate the dataset
Z = reshape(cat(data_list..., dims=1), :, 192);
X_train = Matrix(Z[1:2:end, :])';
X_test = Matrix(Z[2:2:end, :])';

labels_test = labels[2:2:end]

for ndim ∈ 8:32
    M = fit(PCA , X_train; maxoutdim=ndim);
    #Y_train = predict(M, X_test);
    Y_test = predict(M, X_test);
    #Y_all = hcat(Y_test, Y_train);
    X_rec  = reconstruct(M, Y_test);

    K = kmeans(Y_test, 2; maxiter=100) 

    cm = ConfusionMatrix(labels_test .- 1, assignments(K) .- 1)
    foo = [cm.tp cm.fp; cm.tn cm.fn]

    p = heatmap([cm.tp cm.fp; cm.tn cm.fn], palette=cgrad(:algae),
                title=string("ndim = ", ndim))
    annotate!(p, [(1.0,  2.0, (string("TP: ", cm.tp), 16, :green)),
                  (2.0, 2.0, (string("FP: ", cm.fp), 16, :green)),
                  (1.0, 1.0, (string("TN: " , cm.tn), 16, :green)),
                  (2.0, 1.0, (string("FN: ", cm.fn),  16, :green))])
    fname = @sprintf "PCA_ndim%02d.png"  ndim
    savefig(p, fname)
end
