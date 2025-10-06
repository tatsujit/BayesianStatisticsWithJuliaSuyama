using DrWatson
@quickactivate "BayesianStatisticsWithJuliaSuyama"
program_name = "p093-negative-binomial"
using CairoMakie
using LaTeXStrings
using Distributions

################################################################
# NegativeBinomial
################################################################
r = 10
μ = 0.3
d = NegativeBinomial(r, μ)

# mean(X, dims=2)
mean(X, dims=1) # always 3.333... (x*3 = 10)
mean(d)

################################################################
# plot init
################################################################
fig = Figure(size = (800, 600))
Label(fig[0, 1],
      "negative Binomial distribution",
      # L"η = %$η,\text{ }x_{\text{init}} = %$x_init",
      fontsize = 16, font = :bold)
ax = Axis(fig[1,1],
          title = "negative Binomial($r, $μ)",
          xlabel = L"x",
          ylabel = "relative frequency",
          # aspect = DataAspect(),
          aspect = 2,
          )
xs = 0:60
max_val = maximum(X)
hist!(ax, X, bins=max_val+1, normalization = :probability)
lines!(ax, xs, pdf.(d, xs), color = :black, label = "prob dist")
axislegend(ax)
# xs = 0:M
# vs = [[x1, x2, M - x1 - x2] for x1 in xs, x2 in xs]
# probabilities = [pdf(d, v) for v in vs]

# # 0から始まる座標を指定
# x = 0:size(probabilities, 2)-1
# y = 0:size(probabilities, 1)-1
# hm = heatmap!(ax, x, y, probabilities)
# Colorbar(fig[1,2], hm, label = "probability")

fig |> display
safesave(plotsdir(program_name * ".pdf"), fig)


