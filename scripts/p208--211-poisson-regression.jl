# disp, save_fig = true, true
disp, save_fig = true, false
# disp, save_fig = false, true
# disp, save_fig = false, false
"""
これを使うとエラーを抑制できる？
"""
function Base.joinpath(s::AbstractString, n::Nothing)
    joinpath(s)
end
using DrWatson
@quickactivate "BayesianStatisticsWithJuliaSuyama"
program_name = "p208--211-poisson-regression"
using CairoMakie
using LaTeXStrings
using Distributions
using Printf
using IceCream
# using ColorSchemes
# using Colors
using Random

include(srcdir("utility_functions.jl"))

rseed = 1234
Random.seed!(rseed)

################################################################
# hyperparameters
################################################################
# max_iter = 2_000
# max_iter = 100
# max_iter = 300
burnin = 500
burnin = 0
max_iter = burnin + 10^3

################################################################
# plot init
################################################################
fig = Figure(
    size = (1200, 900),
    figure_padding = 30,
)

Label(fig[-1, 2:7],
      "Poisson regression with MCMC (iteration = $(max_iter), burnin = $burnin)", #, (μ1, μ2, σ1, σ2) =  ($μ1, $μ2, $σ1, $σ2)",
      fontsize = 20, font = :bold)

################################################################
# data
################################################################
# 入カデータセット
X_obs = [-2, -1.5, 0.5, 0.7, 1.2]
# 出カデータセット
Y_obs = [0, 0, 2, 1, 8]

ax01 = Axis(fig[0, 2], title = "data (N=$(length(X_obs)))", xlabel = L"x", ylabel = L"y")
scatter!(ax01, X_obs, Y_obs, color = :green)

################################################################
# logistic regression
################################################################
# function generate_logistic(X, μ1, μ2, σ1, σ2) #=> Y, f, w1, w2

################################################################
# parameters
################################################################
μ1 = 0
μ2 = 0
σ1 = 1.0
σ2 = 1.0

# log_joint(w, X, Y, μ1, σ1, μ2, σ2) =
#     logpdf(Normal(μ1, σ1), w[1]) +
#     logpdf(Normal(μ2, σ2), w[2]) +
#     sum(logpdf.(Bernoulli.(sig.(w[1] .* X_obs .+ w[2])), Y_obs))
# function log_joint(w, X, Y, μ1, σ1, μ2, σ2)
#     logpdf(Normal(μ1, σ1), w[1]) +
#     logpdf(Normal(μ2, σ2), w[2]) +
#     sum(logpdf.(Bernoulli.(sig.(w[1] .* X_obs .+ w[2])), Y_obs))
# end
# log_joint([1, 1], [1, 2, 3], [0, 0, 1], 0, 1, 0, 1) # -3.3266541165442827
"""
log_joint: ポアソン回帰の対数同時確率
"""
function log_joint(w, X, Y, μ1, σ1, μ2, σ2)
    # 事前分布
    lp_prior = logpdf(Normal(μ1, σ1), w[1]) + logpdf(Normal(μ2, σ2), w[2])
    # 尤度
    lp_likelihood = sum(logpdf.(Poisson.(exp.(w[1] .* X .+ w[2])), Y))
    return lp_prior + lp_likelihood
end

params = (X_obs, Y_obs, μ1, σ1, μ2, σ2)

ulp(w) = log_joint(w, params...)

w_init = randn(2)

# ここでもやはり GMH と HMC の両方を実行して比較する
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

w1_limits = ((-3, 3), (0, 1.0)) # for rotated hist
w2_limits = ((-3, 3), (0, 1.0)) # for rotated hist
w1_limits_yx = (w1_limits[2], w1_limits[1])
w2_limits_yx = (w2_limits[2], w2_limits[1])

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
# log posterior and samples
################################################################
w1_limits = ((-3, 3), (0, 1.0)) # for rotated hist
w2_limits = ((-3, 3), (0, 1.0)) # for rotated hist

w1_limits_s = w1_limits[1]
w2_limits_s = w2_limits[1]
limits_s = (w1_limits_s, w2_limits_s)
w1s = range(w1_limits_s..., 100)
w2s = range(w2_limits_s..., 100)

ax034 = Axis(fig[0, 3:4],
            title = "unnormalized posterior",
            xlabel = L"w_1", ylabel = L"w_2",
             limits = limits_s,
             )
contour!(ax034, w1s, w2s,
         [exp(ulp([w1, w2])) + eps() for w1 in w1s, w2 in w2s],
         colormap=:jet,  # または :rainbow, :turbo
         levels = 10,
         linewidth = 1,
         labels = true,
         )
ax035 = Axis(fig[0, 5],
            title = "samples from posterior (GMH)",
            xlabel = L"w_1", ylabel = L"w_2",
             limits = limits_s,
             )
scatter!(ax035, param_posterior_GMH[1,:], param_posterior_GMH[2,:],
         alpha = 100/max_iter
         )
ax0367 = Axis(fig[0, 6:7],
            title = "samples from posterior (HMC)",
            xlabel = L"w_1", ylabel = L"w_2",
             limits = limits_s,
              )
scatter!(ax0367, param_posterior_HMC[1,:], param_posterior_HMC[2,:],
         alpha = 100/max_iter
         )


################################################################
# predictive dist.
################################################################
limits_xy = ((-2, 2), (0, 15))
xs = range(limits_xy[1]..., 100)

ax32 = Axis(fig[3, 2],
            title = "predictive distributions (GMH)",
            xlabel = L"x", ylabel = L"y",
            limits = limits_xy,
            xautolimitmargin = (0.1, 0.1),  # x軸に10%のマージン # TODO 効いてなさそう
            yautolimitmargin = (0.1, 0.1),  # y軸に10%のマージン
            )
ax334 = Axis(fig[3, 3:4],
            title = "prediction (GMH)",
            xlabel = L"x", ylabel = L"y",
            limits = limits_xy,
            )
ax35 = Axis(fig[3, 5],
            title = "predictive distributions (HMC)",
            xlabel = L"x", ylabel = L"y",
            limits = limits_xy,
            )
ax367 = Axis(fig[3, 6:7],
            title = "prediction (HMC)",
            xlabel = L"x", ylabel = L"y",
            limits = limits_xy,
            )

fs_GMH, fs_HMC = [], []
for i in 1:size(param_posterior_GMH, 2)
    w1g, w2g = param_posterior_GMH[:, i]
    w1h, w2h = param_posterior_HMC[:, i]
    fg(x) = exp(w1g*x+w2g);     fh(x) = exp(w1h*x+w2h)
    push!(fs_GMH, fg.(xs));     push!(fs_HMC, fh.(xs))
    lines!(ax32, xs, fg.(xs), color = (:green, 10/max_iter))
    lines!(ax35, xs, fh.(xs), color = (:green, 10/max_iter))
end

lines!(ax334, xs, mean(fs_GMH), label = "prediction")
lines!(ax367, xs, mean(fs_HMC), label = "prediction")
scatter!(ax334, X_obs, Y_obs, label = "data", color = :blue)
scatter!(ax367, X_obs, Y_obs, label = "data", color = :blue)


################################################################
# layout adjustment
################################################################
col_sizes = [(1, 0.0),
             (2, 0.3), (3, 0.1), (4, 0.1),
             (5, 0.3), (6, 0.1), (7, 0.1),
             ]
n_rows = 4
# row_sizes = [(i, 1/n_rows) for i in 1:n_rows]
row_sizes = [(i, 1/n_rows) for i in 0:n_rows-1]
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

disp && fig |> display
# save_fig && safesave(plotsdir(program_name * "_colorscheme=" * cs * "_.pdf"), fig)
save_fig && safesave(plotsdir(program_name * ".pdf"), fig)
