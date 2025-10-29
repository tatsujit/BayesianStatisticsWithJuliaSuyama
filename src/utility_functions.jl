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
# function n2ij(n::Int, total::Int)::Tuple{Int,Int}
#     n_row, n_col = n2grid(total)
#     row = div(n-1, n_col)+1
#     col = rem(n-1, n_col)+1
#     return (row, col)
# end
"""
    n2ij(n::Int, total::Int; n_row::Union{Int,Nothing}=nothing, n_col::Union{Int,Nothing}=nothing)::Tuple{Int,Int}

線形インデックス n を行列インデックス (i, j) に変換する。
グリッドの次元は n_row, n_col で指定可能。両方とも nothing の場合は n2grid(total) で自動決定。
（両方指定された場合はそのまま使用（ただし n_row * n_col < total の場合、一部の要素がグリッドからはみ出す可能性があることに注意））

# Examples
```julia
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

julia> # 明示的に 2×5 グリッドを指定
julia> map(x -> n2ij(x, 10; n_row=2, n_col=5), 1:10)
10-element Vector{Tuple{Int64, Int64}}:
 (1, 1)
 (1, 2)
 (1, 3)
 (1, 4)
 (1, 5)
 (2, 1)
 (2, 2)
 (2, 3)
 (2, 4)
 (2, 5)
```
"""
function n2ij(n::Int, total::Int; n_row::Union{Int,Nothing}=nothing, n_col::Union{Int,Nothing}=nothing)::Tuple{Int,Int}
    # グリッドサイズの決定
    if isnothing(n_row) && isnothing(n_col)
        # 両方未指定の場合は n2grid で自動決定
        n_row, n_col = n2grid(total)
    elseif !isnothing(n_row) && isnothing(n_col)
        # n_row のみ指定された場合
        n_col = cld(total, n_row)  # ceiling division
    elseif isnothing(n_row) && !isnothing(n_col)
        # n_col のみ指定された場合
        n_row = cld(total, n_col)
    end
    # 両方指定されている場合はそのまま使用

    # 境界チェック
    @assert 1 ≤ n ≤ total "n must be in range [1, $total], got $n"
    @assert n_row * n_col ≥ total "Grid size $(n_row)×$(n_col) = $(n_row*n_col) is too small for $total elements"

    # 線形インデックスから行列インデックスへの変換 (column-major)
    row = div(n-1, n_col) + 1
    col = rem(n-1, n_col) + 1
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
function generate_linear(X, σ, μ1, μ2, σ1, σ2)
    w1 = rand(Normal(μ1, σ1))
    w2 = rand(Normal(μ2, σ2))
    f(x) = w1 * x + w2
    Y = rand.(Normal.(f.(X), σ))
    return Y, f, w1, w2
end
"""
TODO: そのうち metaprogramming で書き直すと良い
propertynames(ColorSchemes) とかやっても magma とか出てこないのでとりあえずあきらめる
"""
function colorscheme(cs)
    if cs == "magma"
        return ColorSchemes.magma.colors # 黄が薄い、range とって 2:end-1 を使うと濃くて良い
    elseif cs == "plasma"
        return ColorSchemes.plasma.colors # 黄が薄い、range とって 2:end-1 を使うとちょっと明るい
    elseif cs == "turbo"
        return ColorSchemes.turbo.colors # 黄緑が薄い、range とって 2:end-1 を使うとちょっと明るい
    elseif cs == "inferno"
        return ColorSchemes.inferno.colors # 黄色が薄い、range とって 2:end-1 を使うとちょっと明るいが秋っぽくて良い
    elseif cs == "viridis"
        return ColorSchemes.viridis.colors # 1, end を使うと黄色が薄い、 range とって 2:end-1 を使うと似かよってくる
    end
end
# function colorscheme(cs)
#     sym = Symbol(cs)
#     if hasproperty(ColorSchemes.ColorSchemes, sym)
#         return getfield(ColorSchemes.ColorSchemes, sym).colors
#     else
#         Base.error("Unknown colorscheme: $cs")
#     end
# end
# function colorscheme(cs)
#     return eval(:(ColorSchemes.$cs)).colors
# end
function sig(x)
    return 1/(1 + exp(-x))
end
