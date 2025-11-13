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
program_name = "p219-221-state-space"
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
unit = 100
mul_width, mul_height = 18, 12
fig = Figure(
    size = (mul_width * unit, mul_height * unit),
    # figure_padding = 30,
    figure_padding = 0,
)

axes = [Axis(fig[i, j], ) for i in 1:mul_height, j in 1:mul_width] # 12 x 18
hidedecorations!.(axes); hidespines!.(axes)

################
# plot title
################
Label(fig[0, 1:mul_width], "state space sequence data analysis", fontsize = 20, font = :bold, )

################################################################
# hyperparameters
################################################################
μ1, μ2 = 0.0, 0.0
σ1, σ2 = 10.0, 10.0
σ11, σ12, σ13 = 1.0, 1.0, 1.0

################################################################
# data
################################################################
# length of data series
N = 20

# dimensionality of observed data
D = 2

# sequence data (#=, =# for concatenating lines)
Y_obs =
    [1.9 0.2 0.1 1.4 0.3 1.3 1.6 1.5 1.6 2.4 #=
  =# 1.7 3.6 2.8 1.6 3.0 2.8 5.1 5.2 6.0 6.4;
     0.1 0.2 0.9 1.5 4.0 5.0 6.3 5.8 6.4 7.5 #=
  =# 6.7 7.6 8.7 8.2 8.5 9.6 8.4 8.4 8.4 9.0]

################
# vis.
################
ax = Axis(fig[1:div(mul_height, 2), 1:div(mul_width, 2)],
          title = "2 dim sequence data",
          xlabel = L"x", ylabel = L"y",
          )
lines!(ax, Y_obs[1, :], Y_obs[2, :], )#color = :blue, markersize = 8)
scatter!(ax, Y_obs[1, :], Y_obs[2, :], color = :blue, markersize = 12)

################################################################
# noise, log-joint, ...
################################################################
################
# state transition
################
# noise on the initial state
σ1 = 100.0
# noise on the state transition
σ_x = 1.0
# noise on the observation
σ_y = 1.0

# log-joint distribution for state transition sequence
@views transition(X, σ1, σ_x, D, N) =
    logpdf(MvNormal(zeros(D), σ1 * I), X[:, 1]) +
    sum([logpdf(MvNormal(X[:, n-1], σ_x * I), X[:, n]) for n in 2:N])

# log-density for observed data
@views observation(X, Y, σ_y, D, N) =
    sum([logpdf(MvNormal(X[:, n], σ_y * I), Y_obs[:, n]) for n in 1:N])

# log-joint distribution for state transition and observed data
log_joint_tmp(X, Y, σ1, σ_x, σ_y, D, N) =
    transition(X, σ1, σ_x, D, N) +
    observation(X, Y, σ_y, D, N)

# make it a function with DN-dimensional vector input
log_joint(X_vec, Y, σ1, σ_x, σ_y, D, N) =
    log_joint_tmp(reshape(X_vec, D, N), Y, σ1, σ_x, σ_y, D, N)
params = (Y_obs, σ1, σ_x, σ_y, D, N)

# unnormalized log-posterior for state transition sequence
ulp(X_vec) = log_joint(X_vec, params...)

################################################################
# HMC for getting state variable samples
# the samples have (DN x max_iter) dimensions
################################################################
# initial state
X_init = randn(D * N)
# sample size
max_iter = 1_000
# execute HMC
samples, num_accepted = inference_wrapper_HMC(log_joint, params, X_init, max_iter = max_iter)
println("acceptance rate: ", @sprintf("%.3f", num_accepted / max_iter))

################################################################
# visualize the results
################################################################

ax2 = Axis(fig[div(mul_height, 2):end, div(mul_width, 2):end],
           title = "state transition sequence samples",
           xlabel = L"x", ylabel = L"y",
           )
scatter!(ax2,
         [1, 2, 3],
         [4, 5, 6],
         color = :red,
         markersize = 4,
         alpha = 0.3,
         )

################################################################
# display and save plot
################################################################

disp && fig |> display
# save_fig && safesave(plotsdir(program_name * "_colorscheme=" * cs * "_.pdf"), fig)
# save_fig && safesave(plotsdir(program_name * ".pdf"), fig)
save_fig && safesave(plotsdir(program_name * "_max_iter=$max_iter" * "_.pdf"), fig)
