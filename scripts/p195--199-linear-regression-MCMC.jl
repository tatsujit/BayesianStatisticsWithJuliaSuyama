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
program_name = "p195--199-linear-regression-MCMC"
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
# struct PosteriorStats

################################################################
# plot init
################################################################
fig = Figure(
    size = (1500, 1000),
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

# t = L"Displacement $\delta(\theta)$"*"\n"*L"and rotation $\omega_\varphi(\theta)$"
# plot([sin cos],pi/2,0.999*pi,title=t, label=[L"$\delta(\theta)$" L"$\omega_\varphi$"], xlabel = "θ (rad)")

# TODO: 複数行の latexstring をどうすれば書けるか。 L"a" * "\n" * L"b" では駄目
# text = L"""
#        $f(x|w_1, w_2) =$
#        $w_1 x + w_2$
#        """,

# data plot
ax21 = Axis(fig[2, 1], title = "function form", xlabel = L"x", ylabel = L"y")
text!(ax21, 0.0, 0.5,
      text = L"f(x|w_1, w_2) =",
      fontsize = 18,      # color = :black,
      space = :relative,  # これが重要
      # align = (:center, :top),      # axis=(;visible=false), # これは axis を自動で作る場合
      # rotation = π/4,
      )
text!(ax21, 0.25, 0.25,
      text = L"w_1 x + w_2)",
      fontsize = 18,      # color = :black,
      space = :relative,  # これが重要
      # align = (:center, :top),      # axis=(;visible=false), # これは axis を自動で作る場合
      # rotation = π/4,
      )
# または既存のAxisで
hidedecorations!(ax21)  # 全て非表示 ## grid=false)  # gridは残す
hidespines!(ax21)       # spineのみ非表示

################################################################
# linear regression with two parameters
################################################################
# @ic "linear regression with two parameters"
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
max_iter = burnin + 10^3

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
# function create_axis!(fig, col, row, param_idx, limits, title_str, is_hist=false)
# function plot_trace!(ax, data, param_idx)
# function plot_hist_with_stats!(ax, data, stats, param_idx)

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
# comparison of samples and
################################################################
w1s = range(-5.0, 5.0, length=100)
w2s = range(-5.0, 5.0, length=100)
limits = ((-4.5, 4.5), (-4.5, 4.5)) # shared with the three plots

ax32 = Axis(fig[3, 2],
             title = "unnormalized posterior",
             xlabel = L"w_1",
             ylabel = L"w_2",
             limits = limits,
             )
contour!(ax32, w1s, w2s,
          (w1, w2) -> exp(ulp([w1, w2])) + eps(),
          levels = 20,
          colormap = :jet,
          linewidth = 1.0,
          alpha = 0.5,
          )
ax334 = Axis(fig[3, 3:4],
             title = "sample from posterior (GMH)",
             xlabel = L"w_1",
             ylabel = L"w_2",
             limits = limits,
             )
scatter!(ax334, pp_GMH_all.data[1,:], pp_GMH_all.data[2,:],
         alpha = 100/max_iter,
         )
ax35 = Axis(fig[3, 5],
             title = "sample from posterior (GMH) burnin excluded",
             xlabel = L"w_1",
             ylabel = L"w_2",
             limits = limits,
             )
scatter!(ax35, pp_GMH_bi.data[1,:], pp_GMH_bi.data[2,:],
         alpha = 100/max_iter,
         )

ax42 = Axis(fig[4, 2],
             title = "unnormalized posterior",
             xlabel = L"w_1",
             ylabel = L"w_2",
             limits = limits,
             )
contour!(ax42, w1s, w2s,
          (w1, w2) -> exp(ulp([w1, w2])) + eps(),
          levels = 20,
          colormap = :jet,
          linewidth = 1.0,
          alpha = 0.5,
          )
ax434 = Axis(fig[4, 3:4],
             title = "sample from posterior (HMC)",
             xlabel = L"w_1",
             ylabel = L"w_2",
             limits = limits,
             )
scatter!(ax434, pp_HMC_all.data[1,:], pp_HMC_all.data[2,:],
         alpha = 100/max_iter,
         )
ax45 = Axis(fig[4, 5],
             title = "sample from posterior (HMC) burnin excluded",
             xlabel = L"w_1",
             ylabel = L"w_2",
             limits = limits,
             )
scatter!(ax45, pp_HMC_bi.data[1,:], pp_HMC_bi.data[2,:],
         alpha = 100/max_iter,
         )

################################################################
# prediction distribution
################################################################
# function draw_predictive_linear_function!(ax, data)

xs = range(-10, 10, length=100)
pred_dist_limits = ((-10, 10), (-10, 10))
ax367 = Axis(fig[3, 6:7],
             title = "predictive distribution (GMH)",
             xlabel = L"x",
             ylabel = L"y",
             limits = pred_dist_limits,
             )
draw_predictive_linear_function!(ax367, pp_GMH_all.data)
ax467 = Axis(fig[4, 6:7],
             title = "predictive distribution (HMC)",
             xlabel = L"x",
             ylabel = L"y",
             limits = pred_dist_limits,
             )
draw_predictive_linear_function!(ax467, pp_HMC_all.data)


################################################################
# layout adjustment
################################################################
col_sizes = [(1, 0.1),
             (2, 0.25), (3, 0.1), (4, 0.1),
             (5, 0.25), (6, 0.1), (7, 0.1),
             ]
n_rows = 4
row_sizes = [(i, 1/n_rows) for i in 1:n_rows]
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
