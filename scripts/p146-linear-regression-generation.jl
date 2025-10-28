# disp, save_fig = true, true
disp, save_fig = true, false
# disp, save_fig = false, true
# disp, save_fig = false, false
using DrWatson
@quickactivate "BayesianStatisticsWithJuliaSuyama"
program_name = "p146-linear-regression-generation"
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
# regressions
################################################################
# X = [-10, -5, 0, 5, 10]
num_samples = 3

# あらかじめ与えるパラメータと入力集合 X
σ = 1.0 # noise magnitude added to the output
μ1, μ2 = 0.0, 0.0
σ1, σ2 = 10.0, 10.0
X = [-1.0, -0.5, 0, 0.5, 1.0]


n_simulation = 9

n_row, n_col = n2grid(n_simulation) # 2, 3

################################################################
# plot init
################################################################
fig = Figure(
    size = (1200, 800),
    figure_padding = 30,
)

x_lower, x_upper = -2, 2
# y_lower, y_upper = x_lower, x_upper
y_lower, y_upper = -40, 20

aspect = 1
# 可視化する範囲
xs = range(x_lower, x_upper, length=100)
# ys = range(y_lower, y_upper, length=100)
limits = ((x_lower, x_upper), (y_lower, y_upper))

cs = "magma"  # カラースキームの名前
if cs == "magma"
    clrs = ColorSchemes.magma.colors # 黄が薄い、range とって 2:end-1 を使うと濃くて良い
elseif cs == "plasma"
    clrs = ColorSchemes.plasma.colors # 黄が薄い、range とって 2:end-1 を使うとちょっと明るい
elseif cs == "turbo"
    clrs = ColorSchemes.turbo.colors # 黄緑が薄い、range とって 2:end-1 を使うとちょっと明るい
elseif cs == "inferno"
    clrs = ColorSchemes.inferno.colors # 黄色が薄い、range とって 2:end-1 を使うとちょっと明るいが秋っぽくて良い
elseif cs == "viridis"
    clrs = ColorSchemes.viridis.colors # 1, end を使うと黄色が薄い、 range とって 2:end-1 を使うと似かよってくる
end

Label(fig[0, 1:n_col],
      "regression function and data generation with colorscheme = $cs", #"\mu=%$μ\text{, }\sigma=%$σ",
      # L"η = %$η,\text{ }x_{\text{init}} = %$x_init",
      fontsize = 16, font = :bold)

idxs = range(1, length(clrs), length = num_samples+2)[2:end-1]
# clr_idxs = Int.(floor.(range(1, length(clrs), length = num_samples))) # 1 や end は色が薄いとかありがちなので避ける
clr_idxs = Int.(floor.(idxs))
colors = clrs[clr_idxs]

# ################################################################
# # calculate and plot
# ################################################################
# for i in 1:n_simulation
#     row, col = n2ij(i, n_simulation)
#     Y, f, w1, w2 = generate_linear(X, σ, μ1, μ2, σ1, σ2)
#     ax = Axis(fig[row, col], limits = limits)
#     lines!(ax, xs, f.(xs), label = "simulated function")
#     scatter!(ax, X, Y, label = "simulated data")
#     axislegend(ax, position = :rt)
# end

################################################################
# calculation
################################################################
Ys, fs, w1s, w2s = [], [], [], []
for i in 1:n_simulation
    Y, f, w1, w2 = generate_linear(X, σ, μ1, μ2, σ1, σ2)
    push!(Ys, Y)
    push!(fs, f)
    push!(w1s, w1)
    push!(w2s, w2)
end
Yss = vcat(Ys...)
maximum_Y, minimum_Y = maximum(Yss), minimum(Yss)
limits = ((x_lower, x_upper), (minimum_Y - 1.0, maximum_Y + 1.0))
################################################################
# plot
################################################################
for i in 1:n_simulation
    row, col = n2ij(i, n_simulation)
    ax = Axis(fig[row, col], limits = limits)
    lines!(ax, xs, fs[i].(xs), label = "simulated function")
    scatter!(ax, X, Ys[i], label = "simulated data")
    w1s[i] ≥ 0 ? position = :lt : position = :lb
    axislegend(ax, position = position)
end

disp && fig |> display
save_fig && safesave(plotsdir(program_name * "_colorscheme=" * cs * "_.pdf"), fig)
