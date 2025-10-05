using DrWatson
@quickactivate "BayesianStatisticsWithJuliaSuyama"
program_name = "p091-poisson"
using CairoMakie
using LaTeXStrings
using Distributions

################################################################
# poisson
################################################################
μ = 2.0
d = Poisson(μ)
x = rand(d)
X = rand(d, 100)
mean(X, dims=2) # [5.0; 2.88; 2.12;;]
mean(X, dims=1) # always 3.333... (x*3 = 10)
mean(d)

################################################################
# plot init
################################################################
fig = Figure(size = (800, 600))
Label(fig[0, 1],
      "Poisson distribution",
      # L"η = %$η,\text{ }x_{\text{init}} = %$x_init",
      fontsize = 16, font = :bold)
ax = Axis(fig[1,1],
          title = "Poisson($μ)",
          xlabel = L"x",
          ylabel = "frequency",
          # aspect = DataAspect(),
          aspect = 2,
          )
max_val = maximum(X)
hist!(ax, X, bins=max_val+1)

# xs = 0:M
# vs = [[x1, x2, M - x1 - x2] for x1 in xs, x2 in xs]
# probabilities = [pdf(d, v) for v in vs]

# # 0から始まる座標を指定
# x = 0:size(probabilities, 2)-1
# y = 0:size(probabilities, 1)-1
# hm = heatmap!(ax, x, y, probabilities)
# Colorbar(fig[1,2], hm, label = "probability")

fig |> display
