export auc, aucpair


"""
    auc(pred::Vector{<:AbstractFloat}, label::Vector{Int})

Compute Area Under the Curve : https://arize.com/blog/what-is-auc/\n
AUC represents the probability of positive-sample-score ≥ negative-sample-score. i.e.\n

            number of (sample⁺, sample⁻) pairs that
            satisfies sample⁺ score ≥ sample⁻ score
    AUC = ───────────────────────────────────────────
              count of (sample⁺, sample⁻) pairs

    if sample⁺ score  > sample⁻ score, then number += 1
    if sample⁺ score == sample⁻ score, then number += 1/2
```
julia> label = [1  , 0  , 1  , 0  , 0  , 0  , 1  , 0  , 0  , 1  ];
julia> pred  = [0.7, 0.8, 0.9, 0.1, 0.3, 0.5, 0.6, 0.2, 0.8, 0.7];
julia> auc(pred, label)
0.75

julia> label = [1  , 1  , 1  , 1  , 1  , 0  , 0  , 0  , 0  , 0  ];
julia> pred  = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
julia> auc(pred, label)
0.5

julia> label = [1  , 1  , 1  , 1  , 1  , 0  , 0  , 0  , 0  , 0  ];
julia> pred  = [0.9, 0.8, 0.7, 0.6, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9];
julia> auc(pred, label)
0.5
```
"""
function auc(pred::Vector{<:AbstractFloat}, label::Vector{Int})
    m = 0.0  # numerator
    M = 0.0  # denominator
    N = length(pred)
    for i in 1:(N-1)
        for j in (i+1):N
            if label[i] ≠ label[j]
                M += 1.0
                if (label[i] > label[j] && pred[i] > pred[j]) ||
                   (label[i] < label[j] && pred[i] < pred[j])
                    m += 1.0
                elseif pred[i] == pred[j]
                    m += 0.5
                end
            end
        end
    end
    return m / M
end


function aucpair(pred::Vector{<:AbstractFloat}, label::Vector{Int})
    m = 0  # numerator
    M = 0  # denominator
    N = length(pred)
    for i in 1:(N-1)
        for j in (i+1):N
            if label[i] ≠ label[j]
                M += 1
                if (label[i] > label[j] && pred[i] > pred[j]) ||
                   (label[i] < label[j] && pred[i] < pred[j])
                    m += 1.0
                elseif pred[i] == pred[j]
                    m += 0.5
                end
            end
        end
    end
    return m , M
end



"""
    auc(positives::Vector{<:AbstractFloat}, negatives::Vector{<:AbstractFloat})

Compute Area Under the Curve : https://arize.com/blog/what-is-auc/\n
AUC represents the probability of positive-sample-score ≥ negative-sample-score. i.e.\n

            number of (sample⁺, sample⁻) pairs that
            satisfies sample⁺ score ≥ sample⁻ score
    AUC = ───────────────────────────────────────────
              count of (sample⁺, sample⁻) pairs

        if sample⁺ score  > sample⁻ score, then number += 1
        if sample⁺ score == sample⁻ score, then number += 1/2
```
julia> auc([0.7,0.9,0.6,0.7], [0.8,0.1,0.3,0.5,0.2,0.8])
0.75

julia> auc([0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5])
0.5

julia> auc([0.9, 0.8, 0.7, 0.6, 0.5], [0.5, 0.6, 0.7, 0.8, 0.9])
0.5
```
"""
function auc(positives::Vector{<:AbstractFloat}, negatives::Vector{<:AbstractFloat})
    cp = length(positives)
    cn = length(negatives)
    c = 0.00000 # numerator
    C = cp * cn # denominator
    for p in positives
        for n in negatives
            if p > n
                c += 1.0
            elseif p == n
                c += 0.5
            end
        end
    end
    return c / C
end


function aucpair(positives::Vector{<:AbstractFloat}, negatives::Vector{<:AbstractFloat})
    cp = length(positives)
    cn = length(negatives)
    c = 0.00000 # numerator
    C = cp * cn # denominator
    for p in positives
        for n in negatives
            if p > n
                c += 1.0
            elseif p == n
                c += 0.5
            end
        end
    end
    return c , C
end
