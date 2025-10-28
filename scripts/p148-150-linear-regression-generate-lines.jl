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
# regression setting 1
################################################################
# X = [-10, -5, 0, 5, 10]
num_samples = 3

# あらかじめ与えるパラメータと入力集合 X
σ = 1.0 # noise magnitude added to the output
μ1s, μ2s = [-20.0, 0.0, 20.0], [-20.0, 0.0, 20.0]
σ1, σ2 = 10.0, 10.0
X = [-1.0, -0.5, 0, 0.5, 1.0]

n_simulation = length(μ1s) * length(μ2s)
n_row, n_col = n2grid(n_simulation) # 3, 3

################################################################
# plot init
################################################################
fig = Figure(
    size = (800, 1200),
    figure_padding = 30,
)

x_lower, x_upper = -2, 2
# y_lower, y_upper = x_lower, x_upper
y_lower, y_upper = -100, 100

aspect = 0.5
# 可視化する範囲
xs = range(x_lower, x_upper, length=100)
# ys = range(y_lower, y_upper, length=100)
limits = ((x_lower, x_upper), (y_lower, y_upper))

# カラースキーム選択
cs = "magma" # 黄が薄い、range とって 2:end-1 を使うと濃くて良い
# cs = "plasma" # 黄が薄い、range とって 2:end-1 を使うとちょっと明るい
# cs = "turbo" # 黄緑が薄い、range とって 2:end-1 を使うとちょっと明るい
# cs = "inferno" # 黄色が薄い、range とって 2:end-1 を使うとちょっと明るいが秋っぽくて良い
# cs = "viridis" # 1, end を使うと黄色が薄い、 range とって 2:end-1 を使うと似かよってくる

clrs = colorscheme(cs)


Label(fig[0, 1:n_col],
      "regression function and data generation with colorscheme = $cs", #"\mu=%$μ\text{, }\sigma=%$σ",
      # L"η = %$η,\text{ }x_{\text{init}} = %$x_init",
      fontsize = 16, font = :bold)

idxs = range(1, length(clrs), length = num_samples+2)[2:end-1]
# clr_idxs = Int.(floor.(range(1, length(clrs), length = num_samples))) # 1 や end は色が薄いとかありがちなので避ける
clr_idxs = Int.(floor.(idxs))
colors = clrs[clr_idxs]

################################################################
# calculation
################################################################
Ys, fss, w1s, w2s = [], [], [], []
for (i, μ1) in enumerate(μ1s), (j, μ2) in enumerate(μ2s)
    # Y, f, w1, w2 = generate_linear(X, σ, μ1, μ2, σ1, σ2)
    fs = [generate_linear(X, σ, μ1, μ2, σ1, σ2) for _ in 1:100]
    push!(fss, [x[2] for x in fs])
end
# Yss = vcat(Ys...)
# maximum_Y, minimum_Y = maximum(Yss), minimum(Yss)
# limits = ((x_lower, x_upper), (minimum_Y - 1.0, maximum_Y + 1.0))

################################################################
# plot
################################################################
for (i, μ1) in enumerate(μ1s), (j, μ2) in enumerate(μ2s)
    idx = (i-1)*length(μ2s) + j # まとめて順序を定める
    ax = Axis(fig[i, j], limits = limits,
              xlabel = "x", ylabel = "y",
              title = "μ1 = $(μ1), μ2 = $(μ2)")
    for f in fss[idx]
        lines!(ax, xs, f.(xs), label = "g", alpha = 0.1)
    end
end


################################################################
# regression setting 2
################################################################
# あらかじめ与えるパラメータと入力集合 X
σ = 1.0 # noise magnitude added to the output
μ1, μ2 = 0.0, 0.0
σ1s, σ2s = [1, 10, 20], [1, 10, 20]

n_simulation = length(σ1s) * length(σ2s)

################################################################
# calculation
################################################################
Ys, fss, w1s, w2s = [], [], [], []
for (i, σ1) in enumerate(σ1s), (j, σ2) in enumerate(σ2s)
    # Y, f, w1, w2 = generate_linear(X, σ, μ1, μ2, σ1, σ2)
    fs = [generate_linear(X, σ, μ1, μ2, σ1, σ2) for _ in 1:100]
    push!(fss, [x[2] for x in fs])
end

################################################################
# plot
################################################################
for (i, σ1) in enumerate(σ1s), (j, σ2) in enumerate(σ2s)
    idx = (i-1)*length(σ2s) + j # まとめて順序を定める
    ax = Axis(fig[n_row + i, j], limits = limits,
              xlabel = "x", ylabel = "y",
              title = "σ1 = $(σ1), σ2 = $(σ2)")
    for f in fss[idx]
        lines!(ax, xs, f.(xs), label = "g", alpha = 0.1)
    end
end


################################################################
# display and save
################################################################
disp && fig |> display
save_fig && safesave(plotsdir(program_name * "_colorscheme=" * cs * "_.pdf"), fig)
