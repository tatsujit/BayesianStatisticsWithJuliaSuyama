# disp, save_fig = true, true
disp, save_fig = true, false
# disp, save_fig = false, true
# disp, save_fig = false, false
using DrWatson
@quickactivate "BayesianStatisticsWithJuliaSuyama"
program_name = "p159-logistic-regression"
using CairoMakie
using LaTeXStrings
using Distributions
using Printf
using IceCream
using ColorSchemes
using Colors
using Random

include(srcdir("utility_functions.jl"))

rseed = 123456789
Random.seed!(rseed)

################################################################
# logistic regression
################################################################
function generate_logistic(X, μ1, μ2, σ1, σ2)
    w1 = rand(Normal(μ1, σ1))
    w2 = rand(Normal(μ2, σ2))
    f(x) = sig(w1 * x + w2)
    Y = rand.(Bernoulli.(f.(X)))
    return Y, f, w1, w2
end

################################################################
# parameters
################################################################
μ1 = 0
μ2 = 0
σ1 = 10.0
σ2 = 10.0
X = [-1.0, -0.5, 0, 0.5, 1.0]

# visualization range
xs = range(-2, 2, length=100)

n_simulation = 9
n_row, n_col = n2grid(n_simulation)

################################################################
# plot init
################################################################
fig = Figure(
    size = (800, 800),
    figure_padding = 30,
)

x_lower, x_upper = -2, 2
y_lower, y_upper = 0, 1

aspect = 1.2

Label(fig[0, 1:n_col],
      "logistic regression, (μ1, μ2, σ1, σ2) =  ($μ1, $μ2, $σ1, $σ2)",
      fontsize = 16, font = :bold)

################################################################
# calc and plot
################################################################
for i in 1:n_simulation
    row, col = n2ij(i, n_simulation)

    Y, f, w1, w2 = generate_logistic(X, μ1, μ2, σ1, σ2)
    w1_str, w2_str = @sprintf("%.2f", w1), @sprintf("%.2f", w2)

    ax = Axis(fig[row, col], title="(w1, w2) =  ($w1_str, $w2_str)")
    lines!(ax, xs, f.(xs), label = "simulated function")
    scatter!(ax, X, Y, label = "simulated data")
    axislegend(ax, position = w1 >= 0 ? :lc : :lc,
               backgroundcolor = (:white, 0.75),
               fontsize = 10)
end

################################################################
# setting row and col labels and sizes
################################################################
# Label(fig[1, 0], "linear", fontsize = 16, font = :bold, rotation=π/2, )
# Label(fig[2, 0], "logistic", fontsize = 16, font = :bold, rotation=π/2, )
# Label(fig[3, 0], "Poisson", fontsize = 16, font = :bold, rotation=π/2, )
# Label(fig[0, 1], "sampled parameters", fontsize = 16, font = :bold)
# Label(fig[0, 2], "functions", fontsize = 16, font = :bold)
# Label(fig[0, 3], "data = function + noise", fontsize = 16, font = :bold)
# Label(fig[0, 4], "data = function + noise", fontsize = 16, font = :bold)
# Label(fig[0, 5], "data = function + noise", fontsize = 16, font = :bold)
# # 列の幅を相対的に設定
# for row in 1:n_row
#     rowsize!(fig.layout, row, Relative(1/n_row))   # 1列目: 1/3
# end
# for col in 1:n_col
#     colsize!(fig.layout, col, Relative(1/n_col))   # 1列目: 1/3
# end


disp && fig |> display
save_fig && safesave(plotsdir(program_name * "_colorscheme=" * cs * "_.pdf"), fig)
