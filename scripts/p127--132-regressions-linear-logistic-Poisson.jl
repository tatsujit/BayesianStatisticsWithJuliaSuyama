# disp, save_fig = true, true
disp, save_fig = true, false
# disp, save_fig = false, true
# disp, save_fig = false, false
using DrWatson
@quickactivate "BayesianStatisticsWithJuliaSuyama"
program_name = "p127--132-regressions-linear-logistic-Poisson"
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

μs = [[0.0, 0.0],
      ]
Σs = [
    [0.1 0.0;
     0.0 0.1],
    [0.01 0.0;
     0.0 0.01],
]
σ = 1.0 # noise magnitude added to the output

# n_row, n_col = n2grid(length(ps))
n_row, n_col = 3, 5

################################################################
# plot init
################################################################
fig = Figure(
    size = (1200, 800),
    figure_padding = 30,
)

x_lower, x_upper = -12, 12
# y_lower, y_upper = x_lower, x_upper
y_lower, y_upper = -6, 6

aspect = 1

xs = range(x_lower, x_upper, length=100)
ys = range(y_lower, y_upper, length=100)
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

Label(fig[-1, 1:n_col],
      "regression function and data generation with colorscheme = $cs", #"\mu=%$μ\text{, }\sigma=%$σ",
      # L"η = %$η,\text{ }x_{\text{init}} = %$x_init",
      fontsize = 16, font = :bold)

idxs = range(1, length(clrs), length = num_samples+2)[2:end-1]
# clr_idxs = Int.(floor.(range(1, length(clrs), length = num_samples))) # 1 や end は色が薄いとかありがちなので避ける
clr_idxs = Int.(floor.(idxs))
colors = clrs[clr_idxs]

################################################################
# setting row and col labels and sizes
################################################################
Label(fig[1, 0], "linear", fontsize = 16, font = :bold, rotation=π/2, )
Label(fig[2, 0], "logistic", fontsize = 16, font = :bold, rotation=π/2, )
Label(fig[3, 0], "Poisson", fontsize = 16, font = :bold, rotation=π/2, )
Label(fig[0, 1], "sampled parameters", fontsize = 16, font = :bold)
Label(fig[0, 2], "functions", fontsize = 16, font = :bold)
Label(fig[0, 3], "data = function + noise", fontsize = 16, font = :bold)
Label(fig[0, 4], "data = function + noise", fontsize = 16, font = :bold)
Label(fig[0, 5], "data = function + noise", fontsize = 16, font = :bold)
# 列の幅を相対的に設定
for row in 1:n_row
    rowsize!(fig.layout, row, Relative(1/n_row))   # 1列目: 1/3
end
for col in 1:n_col
    colsize!(fig.layout, col, Relative(1/n_col))   # 1列目: 1/3
end


################################################################
# linear
################################################################
W = rand(MvNormal(μs[1], Σs[1]), num_samples)
Ys = []
fs = []
row, col = n2ij(1, 15; n_row = n_row, n_col = n_col)
axs = [Axis(fig[1, i], ) for i in 1:5]
for n in 1:num_samples
    w1, w2 = W[:, n]
    scatter!(axs[1], w1, w2, markersize = 20, color = colors[n])
    f(x) = w1*x + w2
    push!(fs, f)
    lines!(axs[2], xs, f.(xs), color = colors[n])
    X = -11:1:11
    Y = rand.(Normal.(f.(X), σ))
    push!(Ys, Y)
    if n == num_samples
        for i in 1:num_samples
            lines!(axs[2+i], xs, fs[i].(xs), color = colors[i])
            scatter!(axs[2+i], X, Ys[i], color = colors[i])
        end
    end
end
################################################################
# logistic
################################################################
sig(x) = 1/(1+exp(-x))
W = rand(MvNormal(μs[1], Σs[2]), num_samples)
Ys = []
fs = []
row, col = n2ij(1, 15; n_row = n_row, n_col = n_col)
axs = [Axis(fig[2, i], ) for i in 1:5]
for n in 1:num_samples
    w1, w2 = W[:, n]
    scatter!(axs[1], w1, w2, markersize = 20, color = colors[n])
    f(x) = sig(w1*x + w2)
    push!(fs, f)
    lines!(axs[2], xs, f.(xs), color = colors[n])
    X = -11:1:11
    Y = rand.(Bernoulli.(f.(X)))
    push!(Ys, Y)
    if n == num_samples
        for i in 1:num_samples
            lines!(axs[2+i], xs, fs[i].(xs), color = colors[i])
            scatter!(axs[2+i], X, Ys[i], color = colors[i])
        end
    end
end
################################################################
# Poisson
################################################################
W = rand(MvNormal(μs[1], Σs[2]), num_samples)
Ys = []
fs = []
row, col = n2ij(1, 15; n_row = n_row, n_col = n_col)
axs = [Axis(fig[3, i], ) for i in 1:5]
for n in 1:num_samples
    w1, w2 = W[:, n]
    scatter!(axs[1], w1, w2, markersize = 20, color = colors[n])
    f(x) = exp(w1*x + w2)
    push!(fs, f)
    lines!(axs[2], xs, f.(xs), color = colors[n])
    X = -11:1:11
    Y = rand.(Poisson.(f.(X)))
    push!(Ys, Y)
    if n == num_samples
        for i in 1:num_samples
            lines!(axs[2+i], xs, fs[i].(xs), color = colors[i])
            scatter!(axs[2+i], X, Ys[i], color = colors[i])
        end
    end
end

disp && fig |> display
save_fig && safesave(plotsdir(program_name * "_colorscheme=" * cs * "_.pdf"), fig)
