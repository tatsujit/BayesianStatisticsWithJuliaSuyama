to_vector(x::AbstractRange) = collect(x)
to_vector(x::Number) = [x]
to_vector(x::AbstractArray) = x

rmse(y, t) = 0.5 * sum((y .- t).^2)
cross_entropy(y, t) = -sum(t .* log.(y .+ 1e-7)) # 0 除算回避

"""
n2grid.(1:10)
#(n, (n_row, n_col), capacity = n_row*n_col)
 (1, (1, 1), 1)
 (2, (1, 2), 2)
 (3, (1, 3), 3)
 (4, (2, 2), 4)
 (5, (2, 3), 6)
 (6, (2, 3), 6)
 (7, (2, 4), 8)
 (8, (2, 4), 8)
 (9, (3, 3), 9)
 (10, (3, 4), 12)
"""
function n2grid(n::Int, verbose=false)::Tuple{Int,Int}
    n_row = floor(Int, sqrt(n))
    n_col = ceil(Int, n / n_row)
    if verbose
        @info "n=$n => (n_row, n_col)=($n_row, $n_col), capacity=$(n_row*n_col)"
    end
    return (n_row, n_col)
end
# n2grids.(1:20)
# map(x -> n2grid(x, true), 1:20)
"""
function n2ij(n::Int, total::Int)::Tuple{Int,Int}

julia> total = 10; map(x -> n2ij(x, total), 1:total)
10-element Vector{Tuple{Int64, Int64}}:
 (1, 1)
 (1, 2)
 (1, 3)
 (1, 4)
 (2, 1)
 (2, 2)
 (2, 3)
 (2, 4)
 (3, 1)
 (3, 2)
"""
function n2ij(n::Int, total::Int)::Tuple{Int,Int}
    n_row, n_col = n2grid(total)
    row = div(n-1, n_col)+1
    col = rem(n-1, n_col)+1
    return (row, col)
end
# total = 10; map(x -> n2ij(x, total), 1:total)
# total = 9; map(x -> n2ij(x, total), 1:total)
# total = 8; map(x -> n2ij(x, total), 1:total)
"""
normize01(x): [0, 1] normalization
"""
function normalize01(x)
    return (x .- minimum(x)) ./ (maximum(x) - minimum(x) + 1e-7)
end
"""
normize11(x): [-1, 1] normalization
"""
function normalize11(x)
    return 2.0 * (x .- minimum(x)) ./ (maximum(x) - minimum(x) + 1e-7) .- 1.0
end
# これ必要じゃない？
# using Statistics
function standardize(x)
    return (x .- mean(x)) ./ (std(x) + 1e-7)
end

"""
    sign_changes(v::AbstractVector{<:Real}) -> Int

実数ベクトルの符号切り替わり回数を返す。
ゼロは無視される。
"""
function sign_changes(v::AbstractVector{<:Real})
    # ゼロでない要素の符号のみを抽出
    signs = sign.(filter(!iszero, v))
    # 符号が変わった回数をカウント
    return count(i -> signs[i] != signs[i+1], 1:length(signs)-1)
end
"""
function gradient_method_1dim(f, x_init, η, maxiter)
1変数関数の最適化
"""
function gradient_method_1dim(f, x_init, η, maxiter)
    # 最適化過程のパラメータを格納する配列
    x_seq = Array{typeof(x_init), 1}(undef, maxiter)
    #  勾配
    df(x) = ForwardDiff.derivative(f, x)
    # 初期値
    x_seq[1] = x_init
    # メインの最適化ループ
    for i in 2:maxiter
        x_seq[i] = x_seq[i-1] + η * df(x_seq[i-1])
    end
    x_seq
end
function approx_integration(x_range, f)
    # 幅
    Δ = x_range[2] - x_range[1]
    # 近似された面積と幅を返す
    area = sum(f(x) * Δ for x in x_range)
    return area, Δ
end
