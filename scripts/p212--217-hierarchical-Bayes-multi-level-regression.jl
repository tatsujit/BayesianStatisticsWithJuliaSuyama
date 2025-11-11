# TODO plot the hyperpriors and priors
# TODO plot class-wise predictions
# TODO p.216 class 1 bigger and flatter → the class-wise predictions
#
disp, save_fig = true, true
# disp, save_fig = true, false
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
program_name = "p212--217-hierarchical-Bayes-multi-level-regression"
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
burnin = 100
# burnin = 500
# burnin = 10^3
# max_iter = burnin + 5 * 10^3
max_iter = burnin + 10^3
# max_iter = burnin + 10^2
# max_iter = burnin + 15^2
# max_iter = burnin + 15^3

################################################################
# plot init
################################################################
fig = Figure(
    size = (1600, 1200),
    figure_padding = 30,
)

Label(fig[-2, 1:7],
      "hierarhical linear regression with MCMC (iteration = $(max_iter), burnin = $burnin)",
      #, (μ1, μ2, σ1, σ2) =  ($μ1, $μ2, $σ1, $σ2)",
      fontsize = 20,
      font = :bold,
      )

colors = [
    colorant"#0072B2",  # Blue
    colorant"#E69F00",  # Orange
    colorant"#56B4E9",  # Sky blue
    colorant"#009E73",  # Green
    colorant"#D55E00",  # Vermilion
]

################
# plot rows
################
title_row = 0
data_and_analytical_row = 1
mcmc_gmh_row = 2
mcmc_hmc_row = 3



################################################################
# parameters
################################################################
μ1 = 0.0
μ2 = 0.0
σ1 = 10.0
σ2 = 10.0
σ11, σ12, σ13 = 1.0, 1.0, 1.0

################################################################
# data
################################################################
# 学習データ
X_obs = [[0.3, 0.4],      # class 1
         [0.2, 0.4, 0.9], # class 2
         [0.6, 0.8, 0.9], # class 3
         ]
Y_obs = [[4.0, 3.7],      # class 1
         [6.0, 7.2, 9.4], # class 2
         [6.0, 6.9, 7.8], # class 3
         ]
n_class = length(X_obs)

################################################################
# data plot
################################################################
ax02 = Axis(fig[-1, 2], title = "data (N=$(length(vec(X_obs))))", xlabel = L"x", ylabel = L"y")
plot_per_class_scatter!(ax02, X_obs, Y_obs, n_class)
axislegend(ax02, position = :lt)



w1, w2 = linear_fit(vcat(Y_obs...), vcat(X_obs...))

# 個別に回帰
w1s = zeros(n_class)
w2s = zeros(n_class)
for i in 1:n_class
    w1_tmp, w2_tmp = linear_fit(Y_obs[i], X_obs[i])
    w1s[i], w2s[i] = w1_tmp, w2_tmp
end

# visualization range
xs = range(0, 1, 100)

lf(x, w1, w2) = w1 * x + w2

ax03 = Axis(fig[-1, 3],
            title = "(a) single regression",
            xlabel = L"x", ylabel = L"y",
             # limits = limits_s,
             )
lines!(ax03, xs, lf.(xs, w1, w2), color = :black, linewidth = 3)
         # [exp(ulp([w1, w2])) + eps() for w1 in w1s, w2 in w2s],
         # colormap=:jet,  # または :rainbow, :turbo
         # levels = 10,
         # linewidth = 1,
         # labels = true,
         # )
ax04 = Axis(fig[-1, 4],
            title = "(b) multiple regression",
            xlabel = L"x", ylabel = L"y",
             # limits = limits_s,
             )

plot_per_class_scatter!(ax03, X_obs, Y_obs, n_class)
plot_per_class_lines!(ax04, xs, lf, w1s, w2s, n_class)
plot_per_class_scatter!(ax04, X_obs, Y_obs, n_class)
# for i in 1:n_class
#     lines!(ax04, xs, lf.(xs, w1s[i], w2s[i]), linewidth = 3)
#     scatter!(ax03, X_obs[i], Y_obs[i], label = "class $i", markersize = 18, )
#     scatter!(ax04, X_obs[i], Y_obs[i], label = "class $i", markersize = 18, )
# end
axislegend(ax02, position = :lt)
axislegend(ax03, position = :lt)
axislegend(ax04, position = :lt)


################################################################
# 対数同時分布の設計
################################################################
@views hyper_prior(w) = logpdf(Normal(μ1, σ1), w[1]) +
    logpdf(Normal(μ2, σ2), w[2])
@views prior(w) = sum(logpdf.(Normal.(w[1], σ11), w[3:5])) +
    sum(logpdf.(Normal.(w[2], σ11), w[6:8]))
@views log_likelihood(Y, X, w) =
    sum([sum(logpdf.(Normal.(Y[i], σ13), w[2+i] .* X[i] .+ w[2+i+3])) for i in 1:n_class])
log_joint(w, X, Y) = hyper_prior(w) + prior(w) + log_likelihood(Y_obs, X_obs, w)
params = (Y_obs, X_obs)
ulp(w) = hyper_prior(w) + prior(w) + log_likelihood(w, params...)

# w_init = [w1, w2, w1_1, w1_2, w1_3, w2_1, w2_2, w2_3]
w_init = randn(8)


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
# plot predictions for each class with GMH
################################################################
axes_prediction_GMH = [
    Axis(fig[0, 2]),
    Axis(fig[0, 3:4]),
    Axis(fig[0, 5]),
]
plot_prediction!(axes_prediction_GMH, param_posterior_GMH, n_class, xs)#; color = (:blue, 0.01))
plot_data!(axes_prediction_GMH, X_obs, Y_obs, n_class; markersize=18)#; color = (:red, 1.0))

################################################################
# plot predictions for each class with HMC
################################################################
axes_prediction_HMC = [
    Axis(fig[4, 2]),
    Axis(fig[4, 3:4]),
    Axis(fig[4, 5]),
]
plot_prediction!(axes_prediction_HMC, param_posterior_HMC, n_class, xs)#; color = (:blue, 0.01))
plot_data!(axes_prediction_HMC, X_obs, Y_obs, n_class; markersize=18)#; color = (:red, 1.0))

################################################################
# Metaprogramming: generate axis configurations
################################################################
const MEAN_COLOR = :red
const MEDIAN_COLOR = :purple

w1_limits = ((-5, 10), (0, 1.5)) # for rotated hist
w2_limits = ((-5, 10), (0, 1.5)) # for rotated hist
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
# predictive dist.
################################################################
limits_xy = ((-2, 2), (0, 15))
xs = range(limits_xy[1]..., 100)

ax32 = Axis(fig[3, 2],
            title = "predictive distributions (GMH)",
            xlabel = L"x", ylabel = L"y",
            limits = limits_xy,
            xautolimitmargin = (0.1, 0.1),  # x軸に10%のマージン # TODO 効いてなさそう
            yautolimitmargin = (0.1, 0.1),  # y軸に10%のマージン # TODO 効いてなさそう
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
    fg(x) = (w1g*x+w2g);     fh(x) = (w1h*x+w2h)
    push!(fs_GMH, fg.(xs));     push!(fs_HMC, fh.(xs))
    lines!(ax32, xs, fg.(xs), color = (:purple, 50/max_iter))
    lines!(ax35, xs, fh.(xs), color = (:purple, 50/max_iter))
    plot_per_class_scatter!(ax32, X_obs, Y_obs, n_class)#; kwargs...)
    plot_per_class_scatter!(ax35, X_obs, Y_obs, n_class)#; kwargs...)
end

lines!(ax334, xs, mean(fs_GMH), label = "prediction", linewidth = 5)
lines!(ax367, xs, mean(fs_HMC), label = "prediction", linewidth = 5)

plot_per_class_scatter!(ax334, X_obs, Y_obs, n_class)
plot_per_class_lines!(ax334, xs, lf, w1s, w2s, n_class)
plot_per_class_scatter!(ax367, X_obs, Y_obs, n_class)
plot_per_class_lines!(ax367, xs, lf, w1s, w2s, n_class)

axislegend(ax334, position = :lt)
axislegend(ax367, position = :lt)
# scatter!(ax334, X_obs, Y_obs, label = "data", color = :blue)
# scatter!(ax367, X_obs, Y_obs, label = "data", color = :blue)


################################################################
# layout adjustment
################################################################
col_sizes = [(1, 0.0),
             (2, 0.3), (3, 0.1), (4, 0.1),
             (5, 0.3), (6, 0.1), (7, 0.1),
             ]
n_rows = 6
# row_sizes = [(i, 1/n_rows) for i in 1:n_rows]
row_sizes = [(i, 1/n_rows) for i in -1:n_rows-2]
map(col_sizes) do cs
    idx, rel_size = cs
    colsize!(fig.layout, idx, Relative(rel_size))
end
map(row_sizes) do rs
    idx, rel_size = rs
    rowsize!(fig.layout, idx, Relative(rel_size))
end

############################################################
# common labels
############################################################

Label(fig[-1, 0], "GMH", fontsize = 16, font = :bold, rotation=π/2, )
Label(fig[3, 0], "HMC", fontsize = 16, font = :bold, rotation=π/2, )
# Label(fig[0, 0], "GMH", fontsize = 16, font = :bold, rotation=π/2, )
# Label(fig[1, 0], "HMC", fontsize = 16, font = :bold, rotation=π/2, )


################################################################
# display and save plot
################################################################

disp && fig |> display
# save_fig && safesave(plotsdir(program_name * "_colorscheme=" * cs * "_.pdf"), fig)
# save_fig && safesave(plotsdir(program_name * ".pdf"), fig)
save_fig && safesave(plotsdir(program_name * "_max_iter=$max_iter" * "_.pdf"), fig)
