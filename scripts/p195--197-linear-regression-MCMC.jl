# TODO 1. こんなに色々まとめて描画処理するなら、構造体でも定義してやる。メタプログラミングも使うべき
# DONE 1. 不要な（縦軸）ラベルを消去
# DONE 2. hist の刻みを綺麗に見せる
# DONE 3. 標準偏差を linestyle = :dash で上下に見せる
#
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
program_name = "p195--linear-regression-MCMC"
using CairoMakie
using LaTeXStrings
using Distributions
using Printf
using IceCream
# using ColorSchemes
# using Colors
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
# output data
Y_obs = [-2.2, -1.0, 1.5]

################################################################
# plot init
################################################################
fig = Figure(
    size = (1500, 500),
    figure_padding = 30,
)

x_lower, x_upper = -2, 2
y_lower, y_upper = 0, 1

aspect = 1.2

Label(fig[0, 1:7],
      "logistic regression (red: mean, purple: median, red dash: mean±std)", #, (μ1, μ2, σ1, σ2) =  ($μ1, $μ2, $σ1, $σ2)",
      fontsize = 20, font = :bold)


# data plot
ax11 = Axis(fig[1, 1], title = "data (N=$(length(X_obs)))", xlabel = L"x", ylabel = L"y")
scatter!(ax11, X_obs, Y_obs, color = :green)

################################################################
# linear regression with two parameters
################################################################
@ic "linear regression with two parameters"
# noise size on y
σ = 1.0

# mean and std of prior distribution
μ1, μ2 = 0.0, 0.0
σ1, σ2 = 10.0, 10.0

log_joint(w, X, Y, σ, μ1, σ1, μ2, σ2) =
    sum(logpdf.(Normal.(w[1] * X .+ w[2], σ), Y)) +
    logpdf(Normal(μ1, σ1), w[1]) +
    logpdf(Normal(μ2, σ2), w[2])
params = (X_obs, Y_obs, σ, μ1, σ1, μ2, σ2)

# unnormalized log-posterior and posterior
ulp(w) = log_joint(w, params...)


################################################################
# MCMC
################################################################
# initial value
w_init = randn(2)

# sampling
max_iter = 2000
burnin = 500
param_posterior_GMH, num_accepted_GMH =
    inference_wrapper_GMH(log_joint, params, w_init,
                          max_iter = max_iter, σ=1.0)
param_posterior_HMC, num_accepted_HMC =
    inference_wrapper_HMC(log_joint, params, w_init,
                          max_iter = max_iter, L=10, ε=1e-1)
# bi: burnin removed
param_posterior_GMH_bi = param_posterior_GMH[:, burnin+1:end]
param_posterior_HMC_bi = param_posterior_HMC[:, burnin+1:end]

ppGMHmeans = mean(param_posterior_GMH, dims=2)
ppGMHmedians = median(param_posterior_GMH, dims=2)
ppGMHstds = std(param_posterior_GMH, dims=2)
ppGMHBmeans = mean(param_posterior_GMH_bi, dims=2)
ppGMHBmedians = median(param_posterior_GMH_bi, dims=2)
ppGMHBstds = std(param_posterior_GMH_bi, dims=2)
ppHMCmeans = mean(param_posterior_HMC, dims=2)
ppHMCmedians = median(param_posterior_HMC, dims=2)
ppHMCstds = std(param_posterior_HMC, dims=2)
ppHMCBmeans = mean(param_posterior_HMC_bi, dims=2)
ppHMCBmedians = median(param_posterior_HMC_bi, dims=2)
ppHMCBstds = std(param_posterior_HMC_bi, dims=2)
mean_color = :red
median_color = :purple

################################################################
# sampling process visualization
################################################################

w1_limits = ((-0.5, 1.5), (0, 3.5)) # rotated, so x and y exchanged
w2_limits = ((-3.0, 1.5), (0, 1.5)) # rotated, so x and y exchanged
w1_limits_yx = (w1_limits[2], w1_limits[1])
w2_limits_yx = (w2_limits[2], w2_limits[1])

# GMH trace plots
ax12 = Axis(fig[1, 2],
            title = L"w_1\text{ sequence (GMH), acceptance rate = }%$(round(num_accepted_GMH/max_iter, sigdigits=2))",
            xlabel = "iteration", ylabel = L"w_1",
            limits = (nothing, w1_limits[1]),
            )
lines!(ax12, param_posterior_GMH[1, :])
ax22 = Axis(fig[2, 2],
            title = L"w_2\text{ sequence (GMH)}",
            xlabel = "iteration", ylabel = L"w_2",
            limits = (nothing, w2_limits[1]),
            yticks = LinearTicks(5), # yticks = WilkinsonTicks(5), # これはなんか動かなかった
            )
lines!(ax22, param_posterior_GMH[2, :])
# GMH hists
ax13 = Axis(fig[1, 3],
            title = "hist",
            xlabel = "prob dens",
            xticks = LinearTicks(3),
            # ylabel = L"w_1", yaxisposition=:right,
            limits = w1_limits_yx,
            )
hist!(ax13, param_posterior_GMH[1, :], direction=:x, normalization=:pdf, bins=50)
hlines!(ax13, [ppGMHmeans[1]], color=mean_color)
hlines!(ax13, [ppGMHmedians[1]], color=median_color)
hlines!(ax13, [ppGMHmeans[1] + ppGMHstds[1], ppGMHmeans[1] - ppGMHstds[1]], color=mean_color, linestyle = :dash)
ax23 = Axis(fig[2, 3],
            title = "hist",
            xlabel = "prob dens",
            # ylabel = L"w_2", yaxisposition=:right,
            limits = w2_limits_yx,
            xticks = LinearTicks(3),
            yticks = LinearTicks(5),
            )
hist!(ax23, param_posterior_GMH[2, :], direction=:x, normalization=:pdf, bins=50)
hlines!(ax23, [ppGMHmeans[2]], color=mean_color)
hlines!(ax23, [ppGMHmedians[2]], color=median_color)
hlines!(ax23, [ppGMHmeans[2] + ppGMHstds[2], ppGMHmeans[2] - ppGMHstds[2]], color=mean_color, linestyle = :dash)
# GMH hists with burnin removed
ax14 = Axis(fig[1, 4],
            title = "hist with burnin=$burnin removed",
            xlabel = "prob dens",
            xticks = LinearTicks(3),
            # ylabel = L"w_1", yaxisposition=:right,
            limits = w1_limits_yx,
            )
hist!(ax14, param_posterior_GMH[1, burnin+1:end], direction=:x, normalization=:pdf, bins=50)
hlines!(ax14, [ppGMHBmeans[1]], color=mean_color)
hlines!(ax14, [ppGMHBmedians[1]], color=median_color)
hlines!(ax14, [ppGMHBmeans[1] + ppGMHBstds[1], ppGMHBmeans[1] - ppGMHBstds[1]], color=mean_color, linestyle = :dash)
ax24 = Axis(fig[2, 4],
            title = "hist with burnin=$burnin removed",
            xlabel = "prob dens",
            # ylabel = L"w_2", yaxisposition=:right,
            limits = w2_limits_yx,
            xticks = LinearTicks(3),
            yticks = LinearTicks(5),
            )
hist!(ax24, param_posterior_GMH[2, burnin+1:end], direction=:x, normalization=:pdf, bins=50)
hlines!(ax24, [ppGMHBmeans[2]], color=mean_color)
hlines!(ax24, [ppGMHBmedians[2]], color=median_color)
hlines!(ax24, [ppGMHBmeans[2] + ppGMHBstds[2], ppGMHBmeans[2] - ppGMHBstds[2]], color=mean_color, linestyle = :dash)

# HMC trace plots
ax15 = Axis(fig[1, 5],
            title = L"w_1\text{ sequence (HMC), acceptance rate = }%$(round(num_accepted_HMC/max_iter, sigdigits=2))",
            xlabel = "iteration", ylabel = L"w_1",
            limits = (nothing, w1_limits[1]),
            )
lines!(ax15, param_posterior_HMC[1, :])
ax25 = Axis(fig[2, 5],
            title = L"w_2\text{ sequence (HMC)}",
            xlabel = "iteration", ylabel = L"w_2",
            limits = (nothing, w2_limits[1]),
            yticks = LinearTicks(5),
            )
lines!(ax25, param_posterior_HMC[2, :])
# HMC hists
ax16 = Axis(fig[1, 6],
            title = "hist",
            xlabel = "prob dens",
            # ylabel = L"w_1",
            xticks = LinearTicks(3),
            limits = w1_limits_yx,
            )
hist!(ax16, param_posterior_HMC[1, :], direction=:x, normalization=:pdf, bins=50)
hlines!(ax16, [ppHMCmeans[1]], color=mean_color)
hlines!(ax16, [ppHMCmedians[1]], color=median_color)
hlines!(ax16, [ppHMCmeans[1] + ppHMCstds[1], ppHMCmeans[1] - ppHMCstds[1]], color=mean_color, linestyle = :dash)
ax26 = Axis(fig[2, 6],
            title = "hist",
            xlabel = "prob dens",
            # ylabel = L"w_2",
            limits = (nothing, w2_limits[1]),
            xticks = LinearTicks(3),
            yticks = LinearTicks(5),
            )
hist!(ax26, param_posterior_HMC[2, :], direction=:x, normalization=:pdf, bins=50)
hlines!(ax26, [ppHMCmeans[2]], color=mean_color)
hlines!(ax26, [ppHMCmedians[2]], color=median_color)
hlines!(ax26, [ppHMCmeans[2] + ppHMCstds[2], ppHMCmeans[2] - ppHMCstds[2]], color=mean_color, linestyle = :dash)
# HMC hists, burnin excluded
ax17 = Axis(fig[1, 7],
            title = "hist with burnin=$burnin considered",
            xlabel = "prob dens",
            xticks = LinearTicks(3),
            # ylabel = L"w_1",
            limits = w1_limits_yx,
            )
hist!(ax17, param_posterior_HMC[1, burnin+1:end], direction=:x, normalization=:pdf, bins=50)
hlines!(ax17, [ppHMCBmeans[1]], color=mean_color)
hlines!(ax17, [ppHMCBmedians[1]], color=median_color)
hlines!(ax17, [ppHMCBmeans[1] + ppHMCBstds[1], ppHMCBmeans[1] - ppHMCBstds[1]], color=mean_color, linestyle = :dash)
ax27 = Axis(fig[2, 7],
            title = "hist with burnin=$burnin considered",
            xlabel = "prob dens",
            # ylabel = L"w_2",
            limits = (nothing, w2_limits[1]),
            xticks = LinearTicks(3),
            yticks = LinearTicks(5),
            )
hist!(ax27, param_posterior_HMC[2, burnin+1:end], direction=:x, normalization=:pdf, bins=50)
hlines!(ax27, [ppHMCBmeans[2]], color=mean_color)
hlines!(ax27, [ppHMCBmedians[2]], color=median_color)
hlines!(ax27, [ppHMCBmeans[2] + ppHMCBstds[2], ppHMCBmeans[2] - ppHMCBstds[2]], color=mean_color, linestyle = :dash)


################################################################
# layout adjustment
################################################################
# 全体の列の幅
colsize!(fig.layout, 1, Relative(0.1))
colsize!(fig.layout, 2, Relative(0.25))
colsize!(fig.layout, 3, Relative(0.1))
colsize!(fig.layout, 4, Relative(0.1))
colsize!(fig.layout, 5, Relative(0.25))
colsize!(fig.layout, 6, Relative(0.1))
colsize!(fig.layout, 7, Relative(0.1))
# 全体の行の高さ
rowsize!(fig.layout, 1, Relative(0.5))
rowsize!(fig.layout, 2, Relative(0.5))

################################################################
# display and save plot
################################################################

disp && fig |> display
# save_fig && safesave(plotsdir(program_name * "_colorscheme=" * cs * "_.pdf"), fig)
save_fig && safesave(plotsdir(program_name * ".pdf"), fig)
