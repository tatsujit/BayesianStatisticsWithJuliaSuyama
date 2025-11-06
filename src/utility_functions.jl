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
    gradient_method_1dim(f, x_init, η, maxiter)
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

using LinearAlgebra
"""
n次元単位行列
"""
# eye(n) = Diagonal{Float64}(I, n) # LoadError: UndefVarError: `Diagonal` not defined
eye(n) = Diagonal{Float64}(I, n)
"""
パラメータ抽出
"""
unzip(a) = map(x -> getfield.(a, x), fieldnames(eltype(a)))

"""
Metropolis-Hastings MCMC with gaussian proposal dist.
"""
function GaussianMH(log_p_tilde, μ0;
                    max_iter::Int=100_000, σ::Float64=1.0,)
    # サンプルを格納する配列
    D = length(μ0)
    # μ_samples = zeros(Float64, D, max_iter)
    μ_samples = Array{typeof(μ0[1]), 2}(undef, D, max_iter)

    # 初期サンプル
    μ_samples[:, 1] = μ0

    # 受容されたサンプルの数
    num_accepted = 1

    for i in 2:max_iter
        # ガウス提案分布から新しいサンプルを生成
        μ_tmp = rand(MvNormal(μ_samples[:, i-1], σ * eye(D)))

        # 受容確率 r を計算
        log_r = (log_p_tilde(μ_tmp) +
            logpdf(MvNormal(μ_tmp, σ * eye(D)), μ_samples[:, i-1])) -
            (log_p_tilde(μ_samples[:, i-1]) +
            logpdf(MvNormal(μ_samples[:, i-1], σ), μ_tmp))
        # accept at prob min(1, r) or not
        is_accepted = min(1, exp(log_r)) > rand()
        new_sample = is_accepted ? μ_tmp : μ_samples[:, i-1]

        # 新しいサンプルを格納
        μ_samples[:, i] = new_sample
        # 受容されたサンプルの数を更新
        num_accepted += is_accepted # Bool は Int に暗黙変換
    end
    return μ_samples, num_accepted
end
function inference_wrapper_GMH(log_joint, params, w_init;
                               max_iter::Int=100_000, σ::Float64=1.0)
    ulp(w) = log_joint(w, params...)
    GaussianMH(ulp, w_init; max_iter=max_iter, σ=σ)
end
using ForwardDiff
"""
Hamiltonian Monte Carlo (HMC) sampling
"""
function HMC(log_p_tilde, μ0;
             max_iter::Int=100_000, L::Int=100, ε::Float64=1e-1)

    # value update by Leapfrog
    function leapfrog(grad, p_in, μ_in, L, ε)
        μ = μ_in
        # μ = μ_in .+ ε * p
        p = p_in + 0.5ε * grad(μ)
        for l in 1:L-1
            μ += ε * p
            p += ε * grad(μ)
        end
        μ += ε * p
        p += 0.5ε * grad(μ)
        return p, μ
    end

    # # 非正規化対数事後分布の勾配関数を計算 # for の中に移動
    # grad(μ) = ForwardDiff.gradient(log_p_tilde, μ)
    # サンプルを格納する配列
    D = length(μ0)
    μ_samples = Array{typeof(μ0[1]), 2}(undef, D, max_iter)
    # 初期サンプル
    μ_samples[:, 1] = μ0
    # 受容されたサンプルの数
    num_accepted = 1

    for i in 2:max_iter
        # 運動量 p の生成
        p_in = randn(size(μ0))
        μ_in = μ_samples[:, i-1]

        # 非正規化対数事後分布の勾配関数を計算
        grad(μ) = ForwardDiff.gradient(log_p_tilde, μ)

        # リープフロッグ法で新しい位置と運動量を計算
        p_out, μ_out = leapfrog(grad, p_in, μ_in, L, ε)

        # 受容確率 r を計算
        log_r = (log_p_tilde(μ_out) +
            logpdf(MvNormal(zeros(D), eye(D)), vec(p_out))) -
            (log_p_tilde(μ_in) +
            logpdf(MvNormal(zeros(D), eye(D)), vec(p_in)))

        # accept at prob min(1, r) or not
        is_accepted = min(1, exp(log_r)) > rand()
        new_sample = is_accepted ? μ_out : μ_in

        # 新しいサンプルを格納
        # @inbounds μ_samples[:, i] = new_sample
        μ_samples[:, i] = new_sample
        # 受容されたサンプルの数を更新
        num_accepted += is_accepted # Bool は Int に暗黙変換
    end
    return μ_samples, num_accepted
end
function inference_wrapper_HMC(log_joint, params, w_init;
                               max_iter::Int=100_000, L::Int=100, ε::Float64=1e-1)
    ulp(w) = log_joint(w, params...)
    HMC(ulp, w_init; max_iter=max_iter, L=L, ε=ε)
end
function gradient_method(f, x_init, η, max_iter)
    g(x) = ForwardDiff.gradient(f, x)
    x_seq = Array{typeof(x_init[1]), 2}(undef, length(x_init), max_iter)
    x_seq[:, 1] .= x_init
    for i in 2:max_iter
        # maximum を探すから - ではなく +
        x_seq[:, i] = x_seq[:, i-1] + η * g(x_seq[:, i-1])
    end
    return x_seq
end

function inference_wrapper_gd(log_joint, params, w_init, η, max_iter)
    ulp(w) = log_joint(w, params...)
    w_seq = gradient_method(ulp, w_init, η, max_iter)
    return w_seq
end
struct PosteriorStats
    data::Matrix{Float64}
    means::Vector{Float64}
    medians::Vector{Float64}
    stds::Vector{Float64}

    function PosteriorStats(data::Matrix{Float64})
        means = vec(mean(data, dims=2))
        medians = vec(median(data, dims=2))
        stds = vec(std(data, dims=2))
        new(data, means, medians, stds)
    end
end
################################################################
# Helper function for axis creation
################################################################
function create_axis!(fig, col, row, param_idx, limits, title_str, is_hist=false)
    options = Dict(
        :title => title_str,
        :xlabel => is_hist ? "prob dens" : (param_idx == 1 ? "iteration" : "iteration"),
        :ylabel => is_hist ? "" : (param_idx == 1 ? L"w_1" : L"w_2"),
        :limits => limits,
    )
    is_hist && (options[:xticks] = LinearTicks(3))
    row == 2 && (options[:yticks] = LinearTicks(5))
    ax = Axis(fig[row, col]; options...)
    return ax
end

function plot_trace!(ax, data, param_idx; burnin_gray=true, legend=true)
    # burnin の背景（一度だけ描画）
    burnin_gray && vspan!(ax, 0.5, burnin + 0.5, color=(:gray, 0.3), label="burnin phase")
    lines!(ax, 1:size(data, 2), data[param_idx, :], linewidth=1.5, label="trace")
    legend && axislegend(ax, position=:rb, backgroundcolor=(:white, 0.5))
end

function plot_hist_with_stats!(ax, data, stats, param_idx)
    hist!(ax, data[param_idx, :], direction=:x, normalization=:pdf, bins=50)
    hlines!(ax, [stats.means[param_idx]], color=MEAN_COLOR)
    hlines!(ax, [stats.medians[param_idx]], color=MEDIAN_COLOR)
    μ_pm_σ = stats.means[param_idx] .+ [-1, 1] .* stats.stds[param_idx]
    hlines!(ax, μ_pm_σ,
            color=MEAN_COLOR, linestyle=:dash)
    # ↓ これだとなぜか new world とか
    # hlines!(ax, stats.means[param_idx] .+ [-1, 1] .* stats.stds[param_idx],
    #         color=MEAN_COLOR, linestyle=:dash)
end
function draw_predictive_linear_function!(ax, data)
    for i in 1:size(data, 2)
        w1, w2 = data[1, i], data[2, i]
        lines!(ax, xs, w1 .* xs .+ w2, color = (:green, 20/max_iter))
    end
end
function generate_logistic(X, μ1, μ2, σ1, σ2) # => Y, f, w1, w2
    w1 = rand(Normal(μ1, σ1))
    w2 = rand(Normal(μ2, σ2))
    f(x) = sig(w1 * x + w2)
    Y = rand.(Bernoulli.(f.(X)))
    return Y, f, w1, w2
end
