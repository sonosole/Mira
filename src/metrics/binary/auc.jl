export auc


"""
    auc(pred::AbstractArray, label::AbstractArray)
compute Area Under the Curve : https://arize.com/blog/what-is-auc/

```
julia> label = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
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
function auc(pred::AbstractArray, label::AbstractArray)
    m = 0  # numerator
    M = 0  # denominator
    N = length(pred)
    for i in 1:(N-1)
        for j in (i+1):N
            if label[i] ≠ label[j]
                M += 1
                if (label[i] > label[j] && pred[i] > pred[j]) ||
                   (label[i] < label[j] && pred[i] < pred[j])
                    m += 1
                elseif pred[i] == pred[j]
                    m += 0.5
                end
            end
        end
    end
    return m / M
end


function aucs2int(pred::AbstractArray, label::AbstractArray)
    m = 0  # numerator
    M = 0  # denominator
    N = length(pred)
    for i in 1:(N-1)
        for j in (i+1):N
            if label[i] ≠ label[j]
                M += 1
                if (label[i] > label[j] && pred[i] > pred[j]) ||
                   (label[i] < label[j] && pred[i] < pred[j])
                    m += 1
                elseif pred[i] == pred[j]
                    m += 0.5
                end
            end
        end
    end
    return m , M
end
