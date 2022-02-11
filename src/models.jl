
"""
Contains functions that return a model when called.
Use those in a training script like so:


using ecei_generative

discriminator = ecei_generative.vanilla_discriminator() |> gpu;
gemerator = ecei_generative.vanilla_generator() |> gpu;

"""

export get_vanilla_discriminator, get_dc_discriminator, get_cat_discriminator,  get_vanilla_generator, get_dc_generator, get_dc_generator_v2


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
                 Conv((3, 7), 32=>32, bias=false), # Image is now 2x6x32
                 BatchNorm(32, act),
                 Flux.flatten,               # Image is now 384 wide 
                 Dropout(0.3),
                 Dense(384, 2),
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
                 BatchNorm(32, leakyrelu),
                 ConvTranspose((3, 5), 32 => 32, bias=false),
                 BatchNorm(32, leakyrelu),
                 ConvTranspose((3, 5), 32 => 32, bias=false),
                 BatchNorm(32, leakyrelu),
                 Conv((3, 5), 32 => 1, pad=SamePad(), tanh))
end



