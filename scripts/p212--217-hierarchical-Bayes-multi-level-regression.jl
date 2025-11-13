# note at [[file:-2021-Juliaで作っ.org]]
#
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
burnin = 100
# burnin = 500
# burnin = 10^3
# max_iter = burnin + 5 * 10^3
max_iter = burnin + 10^3
# max_iter = burnin + 10^2
# max_iter = burnin + 15^2
# max_iter = burnin + 15^3

################################################################
# plot initialization
################################################################
################
# plot init
################
colors = [
    colorant"#0072B2",  # Blue
    colorant"#E69F00",  # Orange
    colorant"#009E73",  # Green
    colorant"#D55E00",  # Vermilion
    colorant"#56B4E9",  # Sky blue
]
fig = Figure(
    size = (1600, 1200),
    # figure_padding = 30,
    figure_padding = 0,
)
################
# plot rows
################
row_label_fontsize = 14
rows = Dict(:title => (0,
                       "hierarhical Bayes linear regression with MCMC (iteration = $(max_iter), burnin = $burnin)",),
            :data => (1,
                      "data and analytical solutions",),
            :mcmc_gmh => (2,
                         "MCMC with GMH",),
            :mcmc_hmc => (3,
                         "MCMC with HMC",),
            :prediction_gmh => (4,
                               "predictions with GMH",),
            :prediction_hmc => (5,
                               "predictions with HMC",),
            :new_prediction_hmc => (6,
                               "HMC pred. w/ class 1 aug",),
            )
for (key, (row_idx, label_text)) in rows
    if key != :title
        Label(fig[row_idx, 1], label_text,
              fontsize = row_label_fontsize,
              font = :bold,
              rotation=4π/9, )
    end
end

################
# plot title
################
Label(fig[rows[:title][1], 1:7], rows[:title][2], fontsize = 20, font = :bold, )

################################################################
# hyperparameters
################################################################
μ1, μ2 = 0.0, 0.0
σ1, σ2 = 10.0, 10.0
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

X_obs_aug = [[0.1, 0.3, 0.4, 0.5, 0.6, 0.9], # class 1
             X_obs[2:3]...] # class 2 and 3
Y_obs_aug = [[4.0, 4.0, 3.7, 3.8, 3.9, 3.7], # class 1
             Y_obs[2:3]...] # class 2 and 3

################################################################
# data plot
################################################################
ax02 = Axis(fig[rows[:data][1], 2], title = "data",
            xlabel = L"x", ylabel = L"y")
plot_per_class_scatter!(ax02, X_obs, Y_obs, n_class)
axislegend(ax02, position = :lt)

# analytical solution (単回帰)
w1, w2 = linear_fit(vcat(Y_obs...), vcat(X_obs...))

# analytical solutions (クラスごとに個別に回帰)
w1s_analytical = zeros(n_class)
w2s_analytical = zeros(n_class)
for i in 1:n_class
    w1_tmp, w2_tmp = linear_fit(Y_obs[i], X_obs[i])
    w1s_analytical[i], w2s_analytical[i] = w1_tmp, w2_tmp
end

# visualization range
xs = range(0, 1, 100)

lf(x, w1, w2) = w1 * x + w2

ax034 = Axis(fig[rows[:data][1], 3:4],
            title = "(a) single regression",
            xlabel = L"x", ylabel = L"y",
             # limits = limits_s,
             )
lines!(ax034, xs, lf.(xs, w1, w2), color = :black, linewidth = 3)

ax05 = Axis(fig[rows[:data][1], 5],
            title = "(b) multiple regression",
            xlabel = L"x", ylabel = L"y",
            )
plot_per_class_scatter!(ax034, X_obs, Y_obs, n_class)
plot_per_class_lines!(ax05, xs, lf, w1s_analytical, w2s_analytical, n_class)
plot_per_class_scatter!(ax05, X_obs, Y_obs, n_class)
axislegend(ax02, position = :lt)
axislegend(ax034, position = :lt)
axislegend(ax05, position = :lt)

################################################################
# TODO plot hyperpriors and priors
################################################################
w1s = range(-20, 20, 100)
w2s = range(-20, 20, 100)
ax06 = Axis(fig[rows[:data][1], 6],
            # title = L"\text{hyperpriors for }w_1\text{ and} w_2",
            title = L"\text{hyperprior for }w_1",
            xlabel = L"w_1", ylabel = L"prob. dens.",
             # limits = limits_s,
             )
lines!(ax06, w1s, pdf.(Normal.(μ1, σ1), w1s))
# lines!(ax06, w2s, pdf.(Normal.(μ2, σ2), w2s))

ax07 = Axis(fig[rows[:data][1], 7],
            title = L"\text{hyperprior for }w_2",
            xlabel = L"w_2", ylabel = "prob. dens.",
             # limits = limits_s,
             )
lines!(ax07, w2s, pdf.(Normal.(μ2, σ2), w2s))


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
# param_posterior: (num_params, num_samples)
# w_init =: μ0 に対応、つまり
# w_init = [w1, w2, w1_1, w1_2, w1_3, w2_1, w2_2, w2_3]
# TODO だから index を変換する Dict を作るべき
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
limits_xy = ((0, 1), (2, 10))
axes_prediction_GMH = [
    Axis(fig[rows[:prediction_gmh][1], 2], limits = limits_xy),
    Axis(fig[rows[:prediction_gmh][1], 3:4], limits = limits_xy),
    Axis(fig[rows[:prediction_gmh][1], 5], limits = limits_xy),
]
plot_prediction!(axes_prediction_GMH, param_posterior_GMH, n_class, xs)#; color = (:blue, 0.01))
plot_data!(axes_prediction_GMH, X_obs, Y_obs, n_class; markersize=18)#; color = (:red, 1.0))

################################################################
# plot predictions for each class with HMC
################################################################
axes_prediction_HMC = [
    Axis(fig[rows[:prediction_hmc][1], 2], limits = limits_xy),
    Axis(fig[rows[:prediction_hmc][1], 3:4], limits = limits_xy),
    Axis(fig[rows[:prediction_hmc][1], 5], limits = limits_xy),
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

# ここでは length(w_init) == 8 あるパラメータについて、どれを使うかプロットするか、は、 param_idx で明示的に指定
# Axis parameters: (col, row, data, stats, with_burnin, method_name, acceptance_rate)
# const AXIS_CONFIG = [
AXIS_CONFIG = [
    # (col, row,            posterior,           trace,      w_limits,        title_suffix, use_burnin, param_idx
    # acceptance_info)
    (2, rows[:mcmc_gmh][1], param_posterior_GMH, pp_GMH_all, (nothing, w1_limits[1]), "GMH", false, 1,
     L"w_1\text{ sequence (GMH), acceptance rate = }%$(round(num_accepted_GMH/max_iter, sigdigits=2))"),
    (3, rows[:mcmc_gmh][1], param_posterior_GMH, pp_GMH_all, w1_limits_yx, "hist", false, 1, "hist", ),
    (4, rows[:mcmc_gmh][1], param_posterior_GMH, pp_GMH_bi, w1_limits_yx, "hist", true, 1, "hist with burnin=$burnin removed"),
    (5, rows[:mcmc_gmh][1], param_posterior_GMH, pp_GMH_all, (nothing, w2_limits[1]), "GMH", false, 2, L"w_2\text{ sequence (GMH)}"),
    (6, rows[:mcmc_gmh][1], param_posterior_GMH, pp_GMH_all, w2_limits_yx, "hist", false, 2, "hist"),
    (7, rows[:mcmc_gmh][1], param_posterior_GMH, pp_GMH_bi, w2_limits_yx, "hist", true, 2, "hist with burnin=$burnin removed"),
    (2, rows[:mcmc_hmc][1], param_posterior_HMC, pp_HMC_all, (nothing, w1_limits[1]), "HMC", false, 1,
     L"w_1\text{ sequence (HMC), acceptance rate = }%$(round(num_accepted_HMC/max_iter, sigdigits=2))", ),
    (3, rows[:mcmc_hmc][1], param_posterior_HMC, pp_HMC_all, w1_limits_yx, "hist", false, 1, "hist"),
    (4, rows[:mcmc_hmc][1], param_posterior_HMC, pp_HMC_bi, w1_limits_yx, "hist", true, 1, "hist with burnin=$burnin considered"),
    (5, rows[:mcmc_hmc][1], param_posterior_HMC, pp_HMC_all, (nothing, w2_limits[1]), "HMC", false, 2, L"w_2\text{ sequence (HMC)}"),
    (6, rows[:mcmc_hmc][1], param_posterior_HMC, pp_HMC_all, w2_limits_yx, "hist", false, 2, "hist"),
    (7, rows[:mcmc_hmc][1], param_posterior_HMC, pp_HMC_bi, w2_limits_yx, "hist", true, 2, "hist with burnin=$burnin considered"),
]
# length(AXIS_CONFIG) # 12
@assert length.([c for c in AXIS_CONFIG]) == fill(9, 12)# 9x12
################################################################
# Generate all axes programmatically
################################################################
generate_axis_and_plot!(fig, AXIS_CONFIG)

################################################################
# predictive dist.
################################################################
# limits_xy = ((-2, 2), (0, 15))
xs = range(limits_xy[1]..., 100)

ax32 = Axis(fig[rows[:prediction_gmh][1], 6],
            title = "predictive distributions (GMH)",
            xlabel = L"x", ylabel = L"y",
            limits = limits_xy,
            # xautolimitmargin = (0.1, 0.1),  # x軸に10%のマージン # TODO 効いてなさそう
            # yautolimitmargin = (0.1, 0.1),  # y軸に10%のマージン # TODO 効いてなさそう
            )
ax334 = Axis(fig[rows[:prediction_gmh][1], 7],
            title = "prediction (GMH)",
            xlabel = L"x", ylabel = L"y",
            limits = limits_xy,
            )
ax35 = Axis(fig[rows[:prediction_hmc][1], 6],
            title = "predictive distributions (HMC)",
            xlabel = L"x", ylabel = L"y",
            limits = limits_xy,
            )
ax367 = Axis(fig[rows[:prediction_hmc][1], 7],
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

plot_per_class_scatter!(ax334, X_obs, Y_obs, n_class)
plot_per_class_lines!(ax334, xs, lf, w1s_analytical, w2s_analytical, n_class)
plot_per_class_scatter!(ax367, X_obs, Y_obs, n_class)
plot_per_class_lines!(ax367, xs, lf, w1s_analytical, w2s_analytical, n_class)

lines!(ax334, xs, mean(fs_GMH), label = "prediction", linewidth = 5)
lines!(ax367, xs, mean(fs_HMC), label = "prediction", linewidth = 5)

axislegend(ax334, position = :lt, backgroundcolor = (:white, 0.5))
axislegend(ax367, position = :lt, backgroundcolor = (:white, 0.5))

################################################################
# classs 1 augmented data, prediction with HMC
################################################################
log_joint_aug(w, X, Y) = hyper_prior(w) + prior(w) + log_likelihood(Y_obs_aug, X_obs_aug, w)
params = (Y_obs_aug, X_obs_aug)
ulp(w) = hyper_prior(w) + prior(w) + log_likelihood(w, params...)

# w_init = [w1, w2, w1_1, w1_2, w1_3, w2_1, w2_2, w2_3]
# w_init = randn(8)
param_posterior_HMC_aug, num_accepted_HMC_aug =
    inference_wrapper_HMC(log_joint_aug, params, w_init,
                          max_iter = max_iter, L=10, ε=1e-1)

# Create posterior stats objects
# pp_HMC_all_aug = PosteriorStats(param_posterior_HMC_aug)
# pp_HMC_bi_aug = PosteriorStats(param_posterior_HMC_aug[:, burnin+1:end])

plot_predictions_per_class!(fig, rows[:new_prediction_hmc][1], [2:2, 3:4, 5:5], "HMC",
                            param_posterior_HMC_aug, X_obs_aug, Y_obs_aug, n_class)


################################################################
# layout adjustment
################################################################
col_sizes = [
    (1, 0.06),
    (2, 0.25), (3, 0.11), (4, 0.11),
    (5, 0.25), (6, 0.11), (7, 0.11),
]
n_rows = length(rows)-1
# row_sizes = [(i, 1/n_rows) for i in 1:n_rows]
row_sizes = vcat([(0, 0.1/1.1)], # 1.1 で割って全体を1.0に収める
                 [(i, 1/n_rows/1.1) for i in 1:n_rows] # 1.1 で割って全体を1.0に収める
                 )
map(col_sizes) do (idx, rel_size)
    # map(col_sizes) do cs
    #     idx, rel_size = cs
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
# save_fig && safesave(plotsdir(program_name * ".pdf"), fig)
save_fig && safesave(plotsdir(program_name * "_max_iter=$max_iter" * "_.pdf"), fig)
