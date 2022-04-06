using Flux
using Flux.Data: DataLoader
using Flux: onehotbatch
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

#wb_logger = WandbLogger(project="ecei_catgan_3class", entity="rkube", config=args)
#np = pyimport("numpy")    


# num_depth is the number of ecei frames bundled together into a single example
data_1 = load_from_hdf(2.6, 2.7, 35000, 50000, "/home/rkube/gpfs/KSTAR/025259", 25259, "GT");
data_2 = load_from_hdf(2.6, 2.7, 5000, 9000, "/home/rkube/gpfs/KSTAR/022289", 22289, "GT");
data_3 = load_from_hdf(2.0, 2.1, 10000, 25000, "/home/rkube/gpfs/KSTAR/025263", 25263, "GR");

# Re-order data_1 and data_2 to have multiple channels per example
num_samples = size(data_1)[end] ÷ args["num_depth"];
data_1 = data_1[:, :, 1:num_samples * args["num_depth"]];
data_1 = reshape(data_1, (24, 8, args["num_depth"], 1, num_samples));

num_samples = size(data_2)[end] ÷ args["num_depth"];
data_2 = data_2[:, :, 1:num_samples * args["num_depth"]];
data_2 = reshape(data_2, (24, 8, args["num_depth"], 1, num_samples));

num_samples = size(data_3)[end] ÷ args["num_depth"];
data_3 = data_3[:, :, 1:num_samples * args["num_depth"]];
data_3 = reshape(data_3, (24, 8, args["num_depth"], 1, num_samples));
data_3[isnan.(data_3)] .= 0f0

data_all = cat(data_1, data_2, data_3, dims=5);

# Scale data_filt to [-1.0; 1.0]
data_all = 2f0 * (data_all .- minimum(data_all)) / (maximum(data_all) - minimum(data_all)) .- 1f0 |> gpu; 

# Label the various classes
labels_1 = onehotbatch(repeat([:a], size(data_1)[end]), [:a, :b, :c]) |> gpu;
labels_2 = onehotbatch(repeat([:b], size(data_2)[end]), [:a, :b, :c]) |> gpu;
labels_3 = onehotbatch(repeat([:c], size(data_3)[end]), [:a, :b, :c]) |> gpu;
labels_all = cat(labels_1, labels_2, labels_3, dims=2);

# Train / test split
split_ratio = 0.8 
num_samples = size(data_all)[5]
num_train = round(split_ratio * num_samples) |> Int 
idx_all = randperm(num_samples);      # Random indices for all samples
idx_train = idx_all[1:num_train];     # Indices for training set
idx_test = idx_all[num_train:end];    # Indices for test set

loader_train = DataLoader((data_all[:, :, :, :, idx_train], labels_all[:, idx_train]), batchsize=args["batch_size"], shuffle=true);
loader_test = DataLoader((data_all[:, :, :, :, idx_test], labels_all[:, idx_test]), batchsize=args["batch_size"], shuffle=true);

model = get_ddc_v1(args) |> gpu;


opt = getfield(Flux, Symbol(args["opt"]))(args["lr"], Tuple(args["beta"]));
ps = Flux.params(model);

all_loss_cs = zeros(length(loader_train) * args["num_epochs"]);
all_loss_simp = zeros(length(loader_train) * args["num_epochs"]);
all_loss_orth = zeros(length(loader_train) * args["num_epochs"]);

iter = 1
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

            @show loss_cs, loss_simp,loss_orth, sigma2
            Zygote.ignore() do  
                all_loss_cs[iter] = loss_cs
                all_loss_simp[iter] = loss_simp 
                all_loss_orth[iter] = loss_orth
            end     
            (loss_cs + loss_simp) / args["num_classes"] + loss_orth
        end         
        grads = back(one(loss))
        Flux.update!(opt, ps, grads)
        iter += 1;
    end     

    # Show class assignments in the batch
    (x,y)  = first(loader_train)
    @show sum(model(x)[end-1:end, :], dims=2)'
end

model_c = model |> cpu;
@save "model.bson" model_c




