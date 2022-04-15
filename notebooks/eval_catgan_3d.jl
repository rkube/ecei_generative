# Evaluate CATGAN model

using ecei_generative
using Plots

using Flux
using Flux: onehot, onecold, onehotbatch
using Flux.Data: DataLoader
using Random
using StatsBase
using LinearAlgebra
using Logging
using JSON
using BSON: @load

using Combinatorics


function transform_dset(data_raw, args)
    num_samples = size(data_raw)[end] ÷ args["num_depth"];
    data_tr = data_raw[:, :, 1:num_samples * args["num_depth"]];
    clamp!(data_tr, -0.15, 0.15);
    trf = fit(UnitRangeTransform, data_tr[:]);
    data_tr = StatsBase.transform(trf, data_tr[:]);
    return reshape(data_tr, (24, 8, args["num_depth"], 1, num_samples));
end


# Find the mapping of data to cluster assignment that maximizes the number of correct assignments across classes
function map_data_to_labels(data_list, labels)
    maxsum = -1
    best_idx = -1
    now_idx = 1
    for perm in permutations(labels)
        # Calculate the sum of correct assignments using this permutation and keep score.
        # Permutation with the largest sum wins
        current_sum = sum([sum(y .== p) for (y,p) in zip(data_list, perm)])

        if current_sum > maxsum
            maxsum = current_sum
            best_idx = now_idx
        end
        now_idx += 1
    end
    cluster_perm = collect(permutations(labels))[best_idx]
end

open("config.json", "r") do io
    global args = JSON.parse(io)
end


# Load basis datasets
# Data sets are shifted 0.1s after the training data
data_1 = load_from_hdf(6.0, 6.1, 35000, 50000, "/home/rkube/gpfs/KSTAR/025260", 25260, "GT"); # 3/2 MI
data_2 = load_from_hdf(5.5, 5.6, 5000, 10000, "/home/rkube/gpfs/KSTAR/025880", 25880, "GR"); # 2/1 MI
data_3 = load_from_hdf(2.7, 2.8, 5000, 9000, "/home/rkube/gpfs/KSTAR/022289", 22289, "GT"); # ELM
data_4 = load_from_hdf(6.0, 6.1, 35000, 50000, "/home/rkube/gpfs/KSTAR/025260", 25260, "GT"); # Another 3/2 MI, not trained on

# Transform all datasets
data_1_tr = transform_dset(data_1, args);
data_2_tr = transform_dset(data_2, args);
data_3_tr = transform_dset(data_3, args);
data_4_tr = transform_dset(data_4, args);

data_all = cat(data_1_tr, data_2_tr, data_3_tr, data_4_tr, dims=5);

# Label the various classes
labels_1 = onehotbatch(repeat([:a], size(data_1_tr)[end]), [:a, :b, :c]);
labels_2 = onehotbatch(repeat([:b], size(data_2_tr)[end]), [:a, :b, :c]);
labels_3 = onehotbatch(repeat([:c], size(data_3_tr)[end]), [:a, :b, :c]);
labels_4 = onehotbatch(repeat([:a], size(data_3_tr)[end]), [:a, :b, :c]);
labels_all = cat(labels_1, labels_2, labels_3, labels_4, dims=2);

loader_1 = DataLoader((data_1_tr, labels_1), batchsize=256, shuffle=true);
loader_2 = DataLoader((data_2_tr, labels_2), batchsize=256, shuffle=true);
loader_3 = DataLoader((data_3_tr, labels_3), batchsize=256, shuffle=true);
loader_4 = DataLoader((data_4_tr, labels_4), batchsize=256, shuffle=true);
loader_all = DataLoader((data_all, labels_all), batchsize=256, shuffle=true);

(x1, y1) = first(loader_1);
(x2, y2) = first(loader_2);
(x3, y3) = first(loader_3);
(x4, y4) = first(loader_4);

num_epochs = 25
cluster_accuracy = zeros(num_epochs);
confusion_matrix = zeros((3, 3, num_epochs));
for epoch ∈ 1:25
    println("================= Epoch $(epoch) =====================")
    @load "/home/rkube/gpfs/catgan_epoch$(epoch).bson" D_c G_c

    # Get predictions for the training data
    ypred_1 = onecold(D_c(x1), [:a, :b, :c]);
    ypred_2 = onecold(D_c(x2), [:a, :b, :c]);
    ypred_3 = onecold(D_c(x3), [:a, :b, :c]);

    # Recover a mapping from dataset to their most likely cluster assignments:
    data_list = [ypred_1, ypred_2, ypred_3]
    cluster_assignments = map_data_to_labels(data_list, [:a, :b, :c])

    # Calculate unsupervised assignment accuracy
    cluster_accuracy[epoch] = sum([sum(y .== assgn) for (y, assgn) in zip(data_list, cluster_assignments)]) / sum(length(y) for y in data_list)
    
    confusion_matrix[:, :, epoch] = reshape(reduce(vcat, [[sum(y .== l) for y in data_list] for l in [:a, :b, :c]]), (3,3))
    # for y ∈ [ypred_1, ypred_2, ypred_3, ypred_4]
    #     preds = map(x -> sum(y .== x), [:a, :b, :c])
    #     @show preds
    # end
end

plot(cluster_accuracy)
heatmap(confusion_matrix[:, :, 20])
