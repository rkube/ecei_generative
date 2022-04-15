# Train Deep-divergence cluster model on ECEi data
# https://arxiv.org/abs/1902.04981

using Flux
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold
using CUDA
using Zygote
using Random
using StatsBase
using LinearAlgebra
using Logging
using JSON
using BSON: @save

using PyCall
using Wandb

using ecei_generative

# Set up logging and pycall

open("config_ddc.json", "r") do io
    global args = JSON.parse(io)
end

np = pyimport("numpy")    

# num_depth is the number of ecei frames bundled together into a single example
data_1 = load_from_hdf(5.9, 6.1, 35000, 50000, "/home/rkube/gpfs/KSTAR/025260", 25260, "GT");
data_2 = load_from_hdf(5.4, 5.6, 5000, 10000, "/home/rkube/gpfs/KSTAR/025880", 25880, "GR");
data_3 = load_from_hdf(2.6, 2.8, 5000, 9000, "/home/rkube/gpfs/KSTAR/022289", 22289, "GT");

data_1_tr = transform_dataset(data_1, args) |> gpu;
data_2_tr = transform_dataset(data_2, args) |> gpu;
data_3_tr = transform_dataset(data_3, args) |> gpu;

# Label the various classes
labels_1 = onehotbatch(repeat([:a], size(data_1_tr)[end]), [:a, :b, :c]) |> gpu;
labels_2 = onehotbatch(repeat([:b], size(data_1_tr)[end]), [:a, :b, :c]) |> gpu;
labels_3 = onehotbatch(repeat([:c], size(data_3_tr)[end]), [:a, :b, :c]) |> gpu;

# Train / test split
split_ratio = 0.5    
# Calculate number of training / validation points from each individual dataset
num_train_1 = round(size(data_1_tr)[end] * split_ratio) |> Int 
num_train_2 = round(size(data_2_tr)[end] * split_ratio) |> Int 
num_train_3 = round(size(data_3_tr)[end] * split_ratio) |> Int 

# One train loader that yields random samples from all data sets
data_train = cat(data_1_tr[:,:,:,:,1:num_train_1], data_2_tr[:,:,:,:,1:num_train_2], data_3_tr[:,:,:,:,1:num_train_3], dims=5);
labels_train = cat(labels_1[:, 1:num_train_1], labels_2[:, 1:num_train_2], labels_3[:, 1:num_train_3], dims=2);
loader_train = DataLoader((data_train, labels_train ), batchsize=args["batch_size"], shuffle=true);

data_test= cat(data_1_tr[:,:,:,:,num_train_1 + 1:end], data_2_tr[:,:,:,:,num_train_2 + 1:end], data_3_tr[:,:,:,:,num_train_3 + 1:end], dims=5);
labels_test = cat(labels_1[:, num_train_1 + 1:end], labels_2[:, num_train_2 + 1:end], labels_3[:, num_train_3 + 1:end], dims=2);
loader_test = DataLoader((data_train, labels_train ), batchsize=args["batch_size"], shuffle=true);

# Loader for verification give samples from individual datasets. These samples are not to be included in loader_train
loader_1 = DataLoader((data_1_tr[:, :, :, :, num_train_1 + 1:end], labels_1[:, num_train_1 + 1:end]), batchsize=args["batch_size"], shuffle=true);
loader_2 = DataLoader((data_2_tr[:, :, :, :, num_train_2 + 1:end], labels_2[:, num_train_2 + 1:end]), batchsize=args["batch_size"], shuffle=true);
loader_3 = DataLoader((data_3_tr[:, :, :, :, num_train_3 + 1:end], labels_3[:, num_train_3 + 1:end]), batchsize=args["batch_size"], shuffle=true);

model = get_ddc_v1(args) |> gpu;

opt = getfield(Flux, Symbol(args["opt"]))(args["lr"], Tuple(args["beta"]));
ps = Flux.params(model);

all_loss_cs = zeros(length(loader_train) * args["num_epochs"]);
all_loss_simp = zeros(length(loader_train) * args["num_epochs"]);
all_loss_orth = zeros(length(loader_train) * args["num_epochs"]);

iter = 1
wb_logger = WandbLogger(project="ecei_ddc_3class_scan1", entity="rkube", config=args)
for epoch ∈ 1:args["num_epochs"]
    @show epoch
    for (x,y) in loader_train
        size(x)[end] != args["batch_size"] && continue
        loss, back = Zygote.pullback(ps) do 
            y_pred = model(x);
            y_hidden = y_pred[1:end - args["num_classes"], :]       # Output of the last fully-connected layer before the softmax
            A = y_pred[end - args["num_classes"] + 1:end, :]            # A contains the cluster assignments. Compared to the article, A is transposed

            # See https://github.com/DanielTrosten/mvc/blob/b0a08fc6c75bdb1fae796f82a7cbfb001bf02047/src/lib/kernel.py#L45
            # Note that Julia's batch dimension is 1 (using 1-based index). PyTorch's batch dimension is 0, counted in 0-based indexing.    
            xyT = y_hidden' * y_hidden
            x2 = sum(y_hidden.^2, dims=1)
            distances_squared = x2' .- 2xyT + repeat(x2, args["batch_size"])
            Zygote.ignore() do
                global sigma2 = 0.15f0 * median(distances_squared);
            end 
            K = exp.(-0.5f0 .* distances_squared / sigma2)

            # Calculate the matrix M.
            xyT = A' * CUDA.CuArray(one(randn(Float32, args["num_classes"], args["num_classes"])))
            x2 = sum(A.^2, dims=1);
            y2 = ones(args["num_classes"])' |> gpu;
            M = exp.(-sqrt.(x2' .- 2xyT + repeat(y2, args["batch_size"]) .+ eps(eltype(x2))))'

            # Linear algebra below corresponds to ∑_{i=1:N-1}_{j=i+1:N}  A[i, :]' * K * A[j, :] / √(A[i, :]' * K * A[i, :] * A[j, :]' * K * A[j, :]))
            # See https://github.com/DanielTrosten/mvc/blob/b0a08fc6c75bdb1fae796f82a7cbfb001bf02047/src/lib/loss.py#L30
            nom_cs = A * K * A';
            dnom_cs = diag(nom_cs) * diag(nom_cs)';
            loss_cs = 2f0 / (args["num_classes"] * (args["num_classes"] -1)) * sum(triu(nom_cs ./ sqrt.(dnom_cs .+ eps(eltype(dnom_cs))), 1))

            nom_simp = M * K * M';
            dnom_simp = diag(nom_simp) * diag(nom_simp)';
            loss_simp =  2f0 / (args["num_classes"] * (args["num_classes"] -1)) * sum(triu(nom_simp ./ sqrt.(dnom_cs .+ eps(eltype(dnom_cs))), 1))          
            loss_orth = 2f0 * sum(triu(A' * A, 1)) / (args["batch_size"] * (args["batch_size"] -1))

            Zygote.ignore() do  
                if (mod(iter, 100) == 0)
                    @show loss_cs, loss_simp,loss_orth, sigma2
                end
                all_loss_cs[iter] = loss_cs
                all_loss_simp[iter] = loss_simp 
                all_loss_orth[iter] = loss_orth
            end     
            (loss_cs + loss_simp) / args["num_classes"] + loss_orth
        end         
        grads = back(one(loss))
        Flux.update!(opt, ps, grads)

        if iter % 100 == 0
            # Get label predictions from test data for each of the 3 different datasets
            pred_list = []
            for loader in [loader_1, loader_2, loader_3]
                (x, _) = first(loader)
                push!(pred_list, onecold(model(x)[end - args["num_classes"] + 1: end, :], [:a, :b, :c]))
            end

            cluster_assignments = map_data_to_labels(pred_list, [:a, :b, :c])
            cluster_accuracy = sum([sum(y .== assgn) for (y, assgn) in zip(pred_list, cluster_assignments)]) / sum(length(y) for y in pred_list)
            @show iter, cluster_accuracy
        end

        global iter += 1;
    end     
end

#model_c = model |> cpu;
#@save "/home/rkube/gpfs/model.bson" model_c




