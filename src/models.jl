
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
    
    # Size annotations are for dim(x)[3] = 10, i.e. num_channels=10
    return Chain(Conv((3, 4, 6), 1=>8, bias=false),         # Image is now 22x5x5x8
                 BatchNorm(8, act),
                 Conv((5, 3, 3), 8=>32, bias=false),        # Image is now 18x3x3x32
                 BatchNorm(32, act),
                 Conv((5, 3, 3), 32=>32, bias=false),       # Image is now 14x1x1x32
                 BatchNorm(32, act),
                 Conv((7, 1, 1), 32=>64, bias=false),       # Image is now 8x1x1x64
                 BatchNorm(64, act),
                 Flux.flatten,
                 Dense(512, 2),
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

    return Chain(Dense(args["latent_dim"], 512, relu, bias=false),
                 Dense(512, 512, act, bias=false),
                 x -> reshape(x, (8, 1, 1, 64, :)),
                 ConvTranspose((7, 1, 1), 64=>32, bias=false),
                 BatchNorm(32, act),
                 ConvTranspose((5, 3, 3), 32=>32, bias=false),
                 BatchNorm(32, act),
                 ConvTranspose((5, 3, 3), 32=>8, bias=false),
                 BatchNorm(8, act),
                 ConvTranspose((3, 4, 6), 8=>1, bias=false))
end

