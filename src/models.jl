
"""
Contains functions that return a model when called.
Use those in a training script like so:


using ecei_generative

discriminator = ecei_generative.vanilla_discriminator() |> gpu;
gemerator = ecei_generative.vanilla_generator() |> gpu;

"""

export get_vanilla_discriminator, get_dc_discriminator, get_cat_discriminator,  get_vanilla_generator, get_dc_generator, get_dc_generator_v2, get_cat_discriminator_3d, get_generator_3d 


function get_vanilla_discriminator(n_features)
    return Chain(Dense(n_features, 256, x -> leakyrelu(x, 0.2f0)),
                 Dropout(0.3),
                 Dense(256, 128, x -> leakyrelu(x, 0.2f0)),
                 Dropout(0.3),
                 Dense(128, 1, sigmoid))
end

function get_dc_discriminator(args)
    if args["activation"] in ["celu", "elu", "leakyrelu"]
        act = Base.Fix2(getfield(NNlib, Symbol(args["activation"])), Float32(args["activation_alpha"]))
    else
        act = getfield(NNlib, Symbol(args["activation"]));
    end
   
    return Chain(Conv((3, 7), 1=>8, bias=false),   # Image is now 6x18x8
                 BatchNorm(8, act),
                 Conv((3, 7), 8=>32, bias=false),  # Image is now 4x12x32
                 BatchNorm(32, act),
                 Conv((3, 7), 32=>32, bias=false), # Image is now 2x6x32
                 BatchNorm(32, act),
                 Flux.flatten,               # Image is now 384 wide 
                 Dropout(0.3),
                 Dense(384, 1, sigmoid));   
end

function get_cat_discriminator(args)
    # Same as dc_discriminator, but a softmax at the end
    if args["activation"] in ["celu", "elu", "leakyrelu"]
        act = Base.Fix2(getfield(NNlib, Symbol(args["activation"])), Float32(args["activation_alpha"]))
    else
        act = getfield(NNlib, Symbol(args["activation"]));
    end
   
    return Chain(Conv((3, 7), 1=>8, bias=false),   # Image is now 6x18x8
                 BatchNorm(8, act),
                 Conv((3, 7), 8=>32, bias=false),  # Image is now 4x12x32
                 BatchNorm(32, act),
                 Conv((3, 7), 32=>64, bias=false), # Image is now 2x6x32
                 BatchNorm(64, act),
                 Flux.flatten,               # Image is now 384 wide 
                 Dropout(0.3),
                 Dense(768, 2),
                 x -> softmax(x));   
end


function get_cat_discriminator_3d(args)
    # Apply 3d convolution
    if args["activation"] in ["celu", "elu", "leakyrelu"]
        act = Base.Fix2(getfield(NNlib, Symbol(args["activation"])), Float32(args["activation_alpha"]))
    else
        act = getfield(NNlib, Symbol(args["activation"]));
    end

    # 4 layers are hard-coded
    # Compile list of the transpose-conv filter sizes for each of the four layers
    filter_size_list = [(args["filter_size_H"][k], args["filter_size_W"][k], args["filter_size_D"][k]) for k in [1,2,3,4]]

    # Calculate the final size of the layers
    final_size_H = reduce((W, S) -> conv_layer_size(W, S...),
                          [(w, 1) for w in args["filter_size_H"]], init=24)
    final_size_W = reduce((W, S) -> conv_layer_size(W, S...),
                          [(w, 1) for w in args["filter_size_W"]], init=8)
    final_size_D = reduce((W, S) -> conv_layer_size(W, S...),
                          [(w, 1) for w in args["filter_size_D"]], init=args["num_depth"])

    final_size = final_size_H * final_size_W * final_size_D * args["num_channels"][4]

    # Size annotations are for dim(x)[3] = 10, i.e. num_channels=10
    return Chain(Conv(filter_size_list[1], 1=>args["num_channels"][1], act, bias=false),
                 #BatchNorm(args["num_channels"][1], act),
                 Conv(filter_size_list[2], args["num_channels"][1] => args["num_channels"][2], act, bias=false),        
                 #BatchNorm(args["num_channels"][2], act),
                 Conv(filter_size_list[3], args["num_channels"][2] => args["num_channels"][3], act, bias=false),     
                 #BatchNorm(args["num_channels"][3], act),
                 Conv(filter_size_list[4], args["num_channels"][3] => args["num_channels"][4], act, bias=false),
                 Flux.flatten, 
                 MinibatchDiscrimination(final_size, args["fc_size"], args["mbatch_hidden"]),
                 #BatchNorm(args["num_channels"][4], act),
                 Dense(final_size + args["fc_size"], args["num_classes"]),
                 x -> softmax(x));
end


function get_vanilla_generator(latent_dim, n_features)return Chain(Dense(latent_dim, 128, x -> leakyrelu(x, 0.2f0)),
                 Dense(128, 128, x -> leakyrelu(x, 0.2f0)),
                 Dense(128, 128, x -> leakyrelu(x, 0.2f0)),
                 Dense(128, n_features, tanh))
end


function get_dc_generator(args)
    if args["activation"] in ["celu", "elu", "leakyrelu"]
        act = Base.Fix2(getfield(NNlib, Symbol(args["activation"])), Float32(args["activation_alpha"]))
    else
        act = getfield(NNlib, Symbol(args["activation"]));
    end

    return Chain(Dense(args["latent_dim"], 384, relu), # Image is now 384 wide
                 x -> reshape(x, (2, 6, 32, :)),  # Image is now 2x6x32
                 ConvTranspose((3, 7), 32 => 32, relu),  # Image is now 4x12x32
                 ConvTranspose((3, 7), 32 => 8, relu),  # Image is now 6x18x8
                 ConvTranspose((3, 7), 8=>1, tanh));
end


function get_dc_generator_v2(args)
    if args["activation"] in ["celu", "elu", "leakyrelu"]
        act = Base.Fix2(getfield(NNlib, Symbol(args["activation"])), Float32(args["activation_alpha"]))
    else
        act = getfield(NNlib, Symbol(args["activation"]));
    end

    return Chain(Dense(args["latent_dim"], 2048, relu, bias=false), 
                 x -> reshape(x, (4, 16, 32, :)), 
                 BatchNorm(32, act),
                 ConvTranspose((3, 5), 32 => 32, bias=false),
                 BatchNorm(32, act),
                 ConvTranspose((3, 5), 32 => 32, bias=false),
                 BatchNorm(32, act),
                 Conv((3, 5), 32 => 1, pad=SamePad(), tanh))
end


function get_generator_3d(args)
    if args["activation"] in ["celu", "elu", "leakyrelu"]
        act = Base.Fix2(getfield(NNlib, Symbol(args["activation"])), Float32(args["activation_alpha"]))
    else
        act = getfield(NNlib, Symbol(args["activation"]));
    end

    # 4 layers are hard-coded
    # Compile list of the transpose-conv filter sizes for each of the four layers
    filter_size_list = [(args["filter_size_H"][k], args["filter_size_W"][k], args["filter_size_D"][k]) for k in [1,2,3,4]]

    # Calculate the final size of the layers
    init_size_H = reduce((W, S) -> conv_layer_size(W, S...),
                          [(w, 1) for w in args["filter_size_H"]], init=24)
    init_size_W = reduce((W, S) -> conv_layer_size(W, S...),
                          [(w, 1) for w in args["filter_size_W"]], init=8)
    init_size_D = reduce((W, S) -> conv_layer_size(W, S...),
                          [(w, 1) for w in args["filter_size_D"]], init=args["num_depth"])

    return Chain(Dense(args["latent_dim"], 512, relu, bias=false),
                 # First layer is from hard-coded 512 to the initial size of the image
                 Dense(512, init_size_H * init_size_W * init_size_D * args["num_channels"][4] * 2, relu, bias=false),
                 # Re-shape to be used as input for transpose convolutions
                 x -> reshape(x, (init_size_H, init_size_W, init_size_D, args["num_channels"][4] * 2, :)),
                 ConvTranspose(filter_size_list[4], args["num_channels"][4] * 2 => 2 * args["num_channels"][3], relu, bias=false),
                 ConvTranspose(filter_size_list[3], args["num_channels"][3] * 2 => 2 * args["num_channels"][2], relu, bias=false),
                 ConvTranspose(filter_size_list[2], args["num_channels"][2] * 2 => 2 * args["num_channels"][1], relu, bias=false),
                 ConvTranspose(filter_size_list[1], args["num_channels"][1] * 2 => 1, tanh, bias=false))
end




function get_ddc(v1)
    # Apply 3d convolution
    if args["activation"] in ["celu", "elu", "leakyrelu"]
        act = Base.Fix2(getfield(NNlib, Symbol(args["activation"])), Float32(args["activation_alpha"]))
    else
        act = getfield(NNlib, Symbol(args["activation"]));
    end

    # 4 layers are hard-coded
    # Compile list of the transpose-conv filter sizes for each of the four layers
    filter_size_list = [(args["filter_size_H"][k], args["filter_size_W"][k], args["filter_size_D"][k]) for k in [1,2,3,4]]

    # Calculate the final size of the layers
    final_size_H = reduce((W, S) -> conv_layer_size(W, S...),
                          [(w, 1) for w in args["filter_size_H"]], init=24)
    final_size_W = reduce((W, S) -> conv_layer_size(W, S...),
                          [(w, 1) for w in args["filter_size_W"]], init=8)
    final_size_D = reduce((W, S) -> conv_layer_size(W, S...),
                          [(w, 1) for w in args["filter_size_D"]], init=args["num_depth"])

    final_size = final_size_H * final_size_W * final_size_D * args["num_channels"][4]

    # Size annotations are for dim(x)[3] = 10, i.e. num_channels=10
    return Chain(Conv(filter_size_list[1], 1=>args["num_channels"][1], act, bias=false),
                 #BatchNorm(args["num_channels"][1], act),
                 Conv(filter_size_list[2], args["num_channels"][1] => args["num_channels"][2], act, bias=false),        
                 #BatchNorm(args["num_channels"][2], act),
                 Conv(filter_size_list[3], args["num_channels"][2] => args["num_channels"][3], act, bias=false),     
                 #BatchNorm(args["num_channels"][3], act),
                 Conv(filter_size_list[4], args["num_channels"][3] => args["num_channels"][4], act, bias=false),
                 Flux.flatten, 
                 Parallel(vcat, x -> x, Chain(Dense(final_size, args["num_classes"], init=Flux.kaiming_uniform), x -> softmax(x))))
end



