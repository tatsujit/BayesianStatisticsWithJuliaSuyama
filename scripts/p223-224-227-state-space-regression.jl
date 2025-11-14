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


N = 20
# input
Z_obs = [10, 10, 10, 10, 10, 10, 10, 10, 10, 15,
         15, 15, 15, 15, 15, 15, 8, 8, 8, 8]
# output
Y_obs = [67, 64, 60, 60, 57, 54, 51, 51, 49, 63,
         62, 62, 58, 57, 53, 51, 24, 22, 23, 19]

σ1 = 10.0
σ_x = 1.0
σ_y = 0.5
σ_w = 100.0

ts = 1:length(Y_obs)

################################################################
# plot init
################################################################
@ic "plot init"
unit_figsize = 100
fig_x, fig_y = 16, 12

fig = Figure(
    size = (fig_x * unit_figsize, fig_y * unit_figsize),
    figure_padding = 30,
)

Label(fig[0, 1:fig_x],
      "aaa",
      # "hierarhical linear regression with MCMC (iteration = $(max_iter), burnin = $burnin)",
      #, (μ1, μ2, σ1, σ2) =  ($μ1, $μ2, $σ1, $σ2)",
      fontsize = 20,
      font = :bold,
      )

colors = [
    colorant"#0072B2",  # Blue
    colorant"#D55E00",  # Vermilion
    colorant"#56B4E9",  # Sky blue
    colorant"#009E73",  # Green
    colorant"#E69F00",  # Orange
]

################################################################
# data plot
################################################################
@ic "data plot"
ax1 = Axis(fig[1:div(fig_y, 2), 1:div(fig_x, 2)], title = "data (N=$(length(vec(Z_obs))))", xlabel = L"z", ylabel = L"y")
plot_per_class_scatter!(ax1, [Z_obs], [Y_obs], 1; # only one class, so []'ed
                        )
axislegend(ax1, position = :lt)

ax2 = Axis(fig[1:div(fig_y, 2), div(fig_x, 2)+1:fig_x], title = "data (N=$(length(vec(Z_obs))))", xlabel = L"z", ylabel = L"y")

# w1, w2 = linear_fit(vcat(Y_obs...), vcat(X_obs...))

# # 個別に回帰
# w1s = zeros(n_class)
# w2s = zeros(n_class)
# for i in 1:n_class
#     w1_tmp, w2_tmp = linear_fit(Y_obs[i], X_obs[i])
#     w1s[i], w2s[i] = w1_tmp, w2_tmp
# end

# # visualization range
# xs = range(0, 1, 100)

# lf(x, w1, w2) = w1 * x + w2

# ax03 = Axis(fig[-1, 3],
#             title = "(a) single regression",
#             xlabel = L"x", ylabel = L"y",
#              # limits = limits_s,
#              )
# lines!(ax03, xs, lf.(xs, w1, w2), color = :black, linewidth = 3)
#          # [exp(ulp([w1, w2])) + eps() for w1 in w1s, w2 in w2s],
#          # colormap=:jet,  # または :rainbow, :turbo
#          # levels = 10,
#          # linewidth = 1,
#          # labels = true,
#          # )
# ax04 = Axis(fig[-1, 4],
#             title = "(b) multiple regression",
#             xlabel = L"x", ylabel = L"y",
#              # limits = limits_s,
#              )

# plot_per_class_scatter!(ax03, X_obs, Y_obs, n_class)
# plot_per_class_lines!(ax04, xs, lf, w1s, w2s, n_class)
# plot_per_class_scatter!(ax04, X_obs, Y_obs, n_class)
# # for i in 1:n_class
# #     lines!(ax04, xs, lf.(xs, w1s[i], w2s[i]), linewidth = 3)
# #     scatter!(ax03, X_obs[i], Y_obs[i], label = "class $i", markersize = 18, )
# #     scatter!(ax04, X_obs[i], Y_obs[i], label = "class $i", markersize = 18, )
# # end
# axislegend(ax02, position = :lt)
# axislegend(ax03, position = :lt)
# axislegend(ax04, position = :lt)

################################################################
# calc
################################################################
@ic "calc"

prior(w, σ_w) = logpdf(MvNormal(zeros(2), σ_w * I), w)

@views transition(X, σ0, σ_x) =
    logpdf(Normal(0, σ0), X[1]) +
    sum(logpdf.(Normal.(X[1:N-1], σ_x), X[2:N]))

@views observation(X, Y, Z, w) =
    sum(logpdf.(Normal.(w[1] * Z .+ w[2] + X, σ_y), Y))

log_joint_tmp(X, w, Y, Z, σ_w, σ0, σ_x) =
    transition(X, σ0, σ_x) +
    observation(X, Y, Z, w) + prior(w, σ_w)
@views log_joint(X_vec, Y, Z, σ_w, σ0, σ_x) =
    transition(X_vec[1:N], σ0, σ_x) +
    observation(X_vec[1:N], Y, Z, X_vec[N+1:N+2]) +
    prior(X_vec[N+1:N+2], σ_w)
σ0 = σ1
params = (Y_obs, Z_obs, σ_w, σ0, σ_x)

# HMC
x_init = randn(N+2)
max_iter = 1_000
# max_iter = 10
samples, num_accepted =
    inference_wrapper_HMC(log_joint, params, x_init,
                          max_iter = max_iter, L=100, ε=1e-2)
println("acceptance_rate = $(num_accepted/max_iter)")

################################################################
# plot inference results
################################################################
@ic "plot inference results"

# explained by regression
for i in 1:max_iter
    w1, w2 = samples[[N+1, N+2], i]
    lines!(ax2, ts, w1 .* Z_obs .+ w2,
           color = (colors[3], alpha_for(10)),
           linewidth = 1,
           # label = i==1 ? "regression lines" : nothing,
           )
end

# explained by state variable
for i in 1:max_iter
    X = samples[1:N, i]
    lines!(ax2, ts, X,
           color = (colors[4], alpha_for(10)),
           linewidth = 1,
           )
end

# the sum of regression and state variable
for i in 1:max_iter
    w1, w2 = samples[[N+1, N+2], i]
    X = samples[1:N, i]
    lines!(ax2, ts, w1 .* Z_obs .+ w2 .+ X,
           color = (colors[5], alpha_for(10)),
           linewidth = 1,
           )
end

# observations
lines!(ax2, ts, Y_obs, color = colors[1], linewidth = 3, label = "observation y")
lines!(ax2, ts, Z_obs, color = colors[2], linewidth = 3, label = "observation z")

# legendのために濃い色の見えない線を追加
lines!(ax2, [NaN], [NaN], color = colors[3], linewidth = 3, label = "regression lines",)
lines!(ax2, [NaN], [NaN], color = colors[4], linewidth = 3, label = "state variable x",)
lines!(ax2, [NaN], [NaN], color = colors[5], linewidth = 3, label = "regression + state variable",)

axislegend(ax2, position = :lt)

################################################################
# display and save plot
################################################################
@ic "display and save plot"

disp && fig |> display
# save_fig && safesave(plotsdir(program_name * "_colorscheme=" * cs * "_.pdf"), fig)
# save_fig && safesave(plotsdir(program_name * ".pdf"), fig)
save_fig && safesave(plotsdir(program_name * "_max_iter=$max_iter" * "_.pdf"), fig)

@ic "end"
