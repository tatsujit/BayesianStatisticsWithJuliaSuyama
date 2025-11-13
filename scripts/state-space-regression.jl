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


prior(w, σ_w) = logpdf(MvNormal(zeros(2), σ_w * I), w)


################################################################
# plot init
################################################################
################
# plot rows
################
title_row = 0
data_and_analytical_row = 1
mcmc_gmh_row = 2
mcmc_hmc_row = 3
prediction_gmh_row = 4
prediction_hmc_row = 5

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
    colorant"#D55E00",  # Vermilion
    colorant"#56B4E9",  # Sky blue
    colorant"#009E73",  # Green
]

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
# display and save plot
################################################################
@ic "display and save plot"

disp && fig |> display
# save_fig && safesave(plotsdir(program_name * "_colorscheme=" * cs * "_.pdf"), fig)
# save_fig && safesave(plotsdir(program_name * ".pdf"), fig)
save_fig && safesave(plotsdir(program_name * "_max_iter=$max_iter" * "_.pdf"), fig)

@ic "end"
