using DrWatson
@quickactivate "BayesianStatisticsWithJuliaSuyama"
program_name = "heatmap-dimensions"
using CairoMakie
using LaTeXStrings
using Distributions

# 発散型
colormap = :RdBu, :coolwarm, :seismic

# 連続型
colormap = :viridis, :plasma, :inferno, :cividis

# 知覚的に均一
colormap = :thermal, :deep, :dense

# すべての利用可能なカラーマップを確認
using ColorSchemes
ColorSchemes.colorschemes






################################################################
# plot init
################################################################
fig = Figure(size = (800, 600))
Label(fig[0, 1:4],
      "Heatmap dimensions",
      # L"η = %$η,\text{ }x_{\text{init}} = %$x_init",
      fontsize = 16, font = :bold)

xs = 0:M
ax = Axis(fig[1,1],
          title = "Bernoulli($p)",
          xlabel = L"x",
          ylabel = "probability"
          )
hm = heatmap!(ax, [1 2 3; 4 5 6; 7 8 9])
Colorbar(fig[1,2], hm, label = L"f(x, y)")

fig |> display
