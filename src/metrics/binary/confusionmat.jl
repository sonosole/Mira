export confusionmat
export ConfusionMat
export accuracy
export recall
export precision
export f1score


"""
                         threshold
        C₊ = P₋  +  P₊     │
              ┌────────────┼───────────────┐
              │  P₋  (FN)  │   P₊   (TP)   │
    ────┬─────┴────────────┼────────────┬──┴───►
        │      N₋  (TN)    │   N₊ (FP)  │
        └──────────────────┼────────────┘
                           │       C₋ = N₋  +  N₊
"""
function confusionmat(y, l; threshold::Real=0.5)
    y₊ = y .> threshold  # 预测为正样本的标记序列  0  0  1  1  1  1
    y₋ = .! y₊           # 预测为负样本的标记序列  1  1  0  0  0  0
    l₊ = l .> threshold  # 正样本的标记序列        0  1  1  1  1  0
    l₋ = .! l₊           # 负样本的标记序列        1  0  0  0  0  1

    C  = length(y)      # Total number of samples
    C₊ = sum(l₊)        # Total number of positive samples
    C₋ = C - C₊         # Total number of negative samples

    p₊ = sum(y₊ .& l₊)  # True Positive  (TP)
    n₋ = sum(y₋ .& l₋)  # True Negative  (TN)
    p₋ = C₊ - p₊        # False Negative (FN)
    n₊ = C₋ - n₋        # False Positive (FP)
    return ConfusionMat(p₊, p₋, n₋, n₊)
end                    #TP, FN, TN, FP


"""
                 threshold
                     │
           ┌─────────┼───────────┐
           │    FN   │     TP    │
    ────┬──┴─────────┼────────┬──┴──►
        │     TN     │   FP   │
        └────────────┼────────┘
                     │
"""
mutable struct ConfusionMat
    TP::Int # True Positive
    FN::Int # False Negative
    TN::Int # True Negative
    FP::Int # False Positive
    function ConfusionMat(TP::Int, FN::Int, TN::Int, FP::Int)
        new(TP, FN, TN, FP)
    end
    function ConfusionMat(predicted, label; threshold=0.5)
        c = confusionmat(predicted, label, threshold=threshold)
        new(c.TP, c.FN, c.TN, c.FP)
    end
end


function Base.show(io::IO, x::ConfusionMat)
    print(io, "ConfusionMat(TP=$(x.TP), FN=$(x.FN), TN=$(x.TN), FP=$(x.FP))")
end


function Base.:+(x::ConfusionMat, y::ConfusionMat)
    return ConfusionMat(x.TP + y.TP,
                        x.FN + y.FN,
                        x.TN + y.TN,
                        x.FP + y.FP)
end


"""
    accuracy(x::ConfusionMat) -> (TP + TN) / (TP + FN + TN + FP)
"""
function accuracy(x::ConfusionMat)
    return (x.TP + x.TN) / (x.TP + x.FN + x.TN + x.FP)
end


"""
    recall(x::ConfusionMat; type::String="+") -> TP / (TP + FN)
    also called true-positive-rate / sensitivity for positive class
    while true-negative-rate / specificity for negative class
"""
function recall(x::ConfusionMat; type::String="+")
    if type == "+"
        return x.TP / (x.TP + x.FN)
    end
    if type == "-"
        return x.TN / (x.TN + x.FP)
    end
    @error "type is + or -, but got $type"
end


"""
    precision(x::ConfusionMat; type::String="+") -> TP / (TP + FP)
"""
function Base.precision(x::ConfusionMat; type::String="+")
    if type == "+"
        return x.TP / (x.TP + x.FP)
    end
    if type == "-"
        return x.TN / (x.TN + x.FN)
    end
    @error "type is + or -, but got $type"
end


"""
    f1score(x::ConfusionMat; type::String="+") -> F₁

                  2
    F₁ = ───────────────────────
            1             1
         ───────  +  ───────────
         recall       precision
"""
function f1score(x::ConfusionMat; type::String="+")
    r = recall(x, type=type)
    p = precision(x, type=type)
    return 2 * p * r / (p + r)
end


"""
    fbetascore(x::ConfusionMat; beta=1.0, type::String="+") -> Fᵦ

                      1
    Fᵦ = ────────────────────────────────
            1     β²         1       1
         ────── ────── + ───────── ──────
         recall 1 + β²   precision 1 + β²

    recall is more important when β > 1
"""
function fscore(x::ConfusionMat; beta::Real=1.0, type::String="+")
    r = recall(x, type=type)
    p = precision(x, type=type)
    β² = beta^2
    return (1 + β²) * p * r / (β² * p + r)
end
