using Statistics

export H_of_p, E_of_H_of_p

#marginalized entropy, Eq. (6)
function H_of_p(y)
    # Calculate Η_X[p(y|D)]. 
    # Average over the probability of the different examples in the current mini-batch
    mm = mean(y, dims=2)
    # Calculate the entropy by summing by summing over the different classes that y can assume
    -sum(mm .* log.(mm .+ eps(eltype(y))))
end

# Conditional entropy, Eqs. (4) and (5)
function E_of_H_of_p(y)
    # Calculate expectation value of the entropy. Here we first calculate the entropy *capital Eta* Η
    # for individual samples before averaging.
    H = -y .* log.(y .+ eps(eltype(y)))
    # Now sum over the class dimension and average over the sample dimension
    sum(H) / size(H, 2)
end
