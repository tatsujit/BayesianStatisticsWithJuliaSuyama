# 2025年11月5日 Claude Haiku 4.5 にリファクターさせた (Opus 4.1よりよかった)。
# その後手動でデバッグして動くようにした
#
# 参考になるのは：
#   - struct PosteriorStats はコンストラクタで各統計量の計算までできるので気軽に定義すべき
#   - const MEAN_COLOR とかは、変数定義が見やすくなるので良い。また、関数から使うなら const にすると速くなるわけだし
#   - const AXIS_CONFIG = [(a, b,), (c, d,), ...] というのも便利ではあるが、
#     ``Axis parameters: (col, row, data, stats, with_burnin, method_name, acceptance_rate)``
#     みたいなのをコピペして使うことになる。
#     _こちらも struct にしてしまって @unpack マクロを使った方がわかりやすいし、必要な変数だけ使えるから良いかと思った_
#   - ベクトルの書き方として
#     [1+1, 1-1] == [1, 1] .+ [-1, 1] .* 1
#     のようにまとめられる。

# disp, save_fig, benchmark_flag = true, true, true
# disp, save_fig, benchmark_flag = true, false, true
# disp, save_fig, benchmark_flag = true, false, false
disp, save_fig, benchmark_flag = true, true, false

"""
これを使うとエラーを抑制できる？
"""
function Base.joinpath(s::AbstractString, n::Nothing)
    joinpath(s)
end

using DrWatson
@quickactivate "BayesianStatisticsWithJuliaSuyama"
program_name = "p195--linear-regression-MCMC-refactored-and-debugged"
using CairoMakie
using LaTeXStrings
using Distributions
using Printf
using IceCream
using Random
using ForwardDiff
using LinearAlgebra
using BenchmarkTools: @btime, @benchmark
import ProgressMeter: Progress, next!, @timed, @showprogress

include(srcdir("utility_functions.jl"))

rseed = 1234
Random.seed!(rseed)

# input data
X_obs = [-2, 1, 5]
Y_obs = [-2.2, -1.0, 1.5]

################################################################
# Data structure for posterior statistics
################################################################
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
# plot init
################################################################
fig = Figure(
    size = (1500, 500),
    figure_padding = 30,
)

w1_limits = ((-0.5, 1.5), (0, 3.5))
w2_limits = ((-3.0, 1.5), (0, 1.5))
w1_limits_yx = (w1_limits[2], w1_limits[1])
w2_limits_yx = (w2_limits[2], w2_limits[1])

Label(fig[0, 1:7],
      "logistic regression (red: mean, purple: median, red dash: mean±std)",
      fontsize = 20, font = :bold)

# data plot
ax11 = Axis(fig[1, 1], title = "data (N=$(length(X_obs)))", xlabel = L"x", ylabel = L"y")
scatter!(ax11, X_obs, Y_obs, color = :green)

# data plot
ax21 = Axis(fig[2, 1], title = "function form", xlabel = L"x", ylabel = L"y")
text!(ax21, 0, 0.5,
      text = L"f(x|w_1, w_2) = " * "\n" *
          L"w_1 x + w_2",
      # fontsize = 18,
      # color = :black,
      space = :relative,  # これが重要
      # align = (:center, :top),
      # axis=(;visible=false), # これは axis を自動で作る場合
      )
# または既存のAxisで
hidedecorations!(ax21)  # 全て非表示 ## grid=false)  # gridは残す
hidespines!(ax21)       # spineのみ非表示

################################################################
# linear regression with two parameters
################################################################
@ic "linear regression with two parameters"
σ = 1.0
μ1, μ2 = 0.0, 0.0
σ1, σ2 = 10.0, 10.0

log_joint(w, X, Y, σ, μ1, σ1, μ2, σ2) =
    sum(logpdf.(Normal.(w[1] * X .+ w[2], σ), Y)) +
    logpdf(Normal(μ1, σ1), w[1]) +
    logpdf(Normal(μ2, σ2), w[2])

params = (X_obs, Y_obs, σ, μ1, σ1, μ2, σ2)
ulp(w) = log_joint(w, params...)

################################################################
# MCMC
################################################################
w_init = randn(2)
# max_iter = 2000
burnin = 500
max_iter = burnin + 1000

param_posterior_GMH, num_accepted_GMH =
    inference_wrapper_GMH(log_joint, params, w_init,
                          max_iter = max_iter, σ=1.0)
param_posterior_HMC, num_accepted_HMC =
    inference_wrapper_HMC(log_joint, params, w_init,
                          max_iter = max_iter, L=10, ε=1e-1)

# Create posterior stats objects
pp_GMH_all = PosteriorStats(param_posterior_GMH)
pp_GMH_bi = PosteriorStats(param_posterior_GMH[:, burnin+1:end])
pp_HMC_all = PosteriorStats(param_posterior_HMC)
pp_HMC_bi = PosteriorStats(param_posterior_HMC[:, burnin+1:end])

################################################################
# Metaprogramming: generate axis configurations
################################################################
const MEAN_COLOR = :red
const MEDIAN_COLOR = :purple

# Axis parameters: (col, row, data, stats, with_burnin, method_name, acceptance_rate)
# const AXIS_CONFIG = [
AXIS_CONFIG = [
    # (col, row, posterior, stats, w_limits, title_suffix, use_burnin, acceptance_info)
    (2, 1, param_posterior_GMH, pp_GMH_all, (nothing, w1_limits[1]), "GMH", false, L"w_1\text{ sequence (GMH), acceptance rate = }%$(round(num_accepted_GMH/max_iter, sigdigits=2))"),
    (2, 2, param_posterior_GMH, pp_GMH_all, (nothing, w2_limits[1]), "GMH", false, L"w_2\text{ sequence (GMH)}"),
    (3, 1, param_posterior_GMH, pp_GMH_all, w1_limits_yx, "hist", false, "hist"),
    (3, 2, param_posterior_GMH, pp_GMH_all, w2_limits_yx, "hist", false, "hist"),
    (4, 1, param_posterior_GMH, pp_GMH_bi, w1_limits_yx, "hist", true, "hist with burnin=$burnin removed"),
    (4, 2, param_posterior_GMH, pp_GMH_bi, w2_limits_yx, "hist", true, "hist with burnin=$burnin removed"),
    (5, 1, param_posterior_HMC, pp_HMC_all, (nothing, w1_limits[1]), "HMC", false, L"w_1\text{ sequence (HMC), acceptance rate = }%$(round(num_accepted_HMC/max_iter, sigdigits=2))"),
    (5, 2, param_posterior_HMC, pp_HMC_all, (nothing, w2_limits[1]), "HMC", false, L"w_2\text{ sequence (HMC)}"),
    (6, 1, param_posterior_HMC, pp_HMC_all, w1_limits_yx, "hist", false, "hist"),
    (6, 2, param_posterior_HMC, pp_HMC_all, w2_limits_yx, "hist", false, "hist"),
    (7, 1, param_posterior_HMC, pp_HMC_bi, w1_limits_yx, "hist", true, "hist with burnin=$burnin considered"),
    (7, 2, param_posterior_HMC, pp_HMC_bi, w2_limits_yx, "hist", true, "hist with burnin=$burnin considered"),
]

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

function plot_trace!(ax, data, param_idx)
    lines!(ax, data[param_idx, :])
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

################################################################
# Generate all axes programmatically
################################################################
axes_dict = Dict()

for (col, row, data, stats, limits, method, use_burnin, title_str) in AXIS_CONFIG
    param_idx = row
    key = (col, row)

    # Determine if this is a trace plot or histogram
    is_hist = (col ∈ [3, 4, 6, 7])
    # @ic (fig, col, row, param_idx, limits, title_str, is_hist)
    ax = create_axis!(fig, col, row, param_idx, limits, title_str, is_hist)
    # @ic (ax, data, stats, param_idx, is_hist)
    axes_dict[key] = (ax, data, stats, param_idx, is_hist)

    # Plot content
    if is_hist
        plot_data = use_burnin ? data[:, burnin+1:end] : data
        plot_hist_with_stats!(ax, plot_data, stats, param_idx)
    else
        plot_trace!(ax, data, param_idx)
    end
end

################################################################
# layout adjustment
################################################################
col_sizes = [(1, 0.1),
             (2, 0.25), (3, 0.1), (4, 0.1),
             (5, 0.25), (6, 0.1), (7, 0.1),
             ]
row_sizes = [(1, 0.5), (2, 0.5)]
map(col_sizes) do cs
    idx, rel_size = cs
    colsize!(fig.layout, idx, Relative(rel_size))
end
map(row_sizes) do rs
    idx, rel_size = rs
    rowsize!(fig.layout, idx, Relative(rel_size))
end
################################################################
# display and save plot
################################################################
disp && (fig |> display)
save_fig && (safesave(plotsdir(program_name * ".pdf"), fig))
