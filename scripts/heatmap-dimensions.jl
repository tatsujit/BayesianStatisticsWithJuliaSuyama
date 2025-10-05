using DrWatson
@quickactivate "BayesianStatisticsWithJuliaSuyama"
program_name = "heatmap-dimensions"
using CairoMakie
using LaTeXStrings
using Distributions

################################################################
# plot init
################################################################
fig = Figure(size = (1000, 600))
Label(fig[0, 1:5],
      "Heatmap dimensions",
      # L"η = %$η,\text{ }x_{\text{init}} = %$x_init",
      fontsize = 16, font = :bold)

A = [1 2 3 4;
     5 6 7 8;
     9 10 11 12]

################################################################
# heatmap!(ax, A)
################################################################
ax = Axis(fig[1, 1],
          aspect = DataAspect(),
          title = "heatmap!(ax, A)",
          )
hm = heatmap!(ax, A, colormap = :viridis)
Colorbar(fig[1:2,2], hm,
         # label = "value",
         )
################################################################
# heatmap!(ax, A')
################################################################
ax = Axis(fig[2, 1],
          aspect = DataAspect(),
          title = "heatmap!(ax, A')",
          )
hm = heatmap!(ax, A', colormap = :viridis)

################################################################
# Julia source 上での行列
################################################################
ax = Axis(fig[1, 3],
          aspect = DataAspect(),
          title = "matrix in Julia source",
          limits = ((0.5, size(A, 2)+1.5), (0.5, size(A, 1)+1.5)),
          )
# 各セルに数値を表示
n_row = size(A, 1)
n_col = size(A, 2)
text!(ax, 1, n_row,
      text = "A = ",
      # align = (:center, :bottom),
      color = :black,
      fontsize = 16)
for i in 1:n_row, j in 1:n_col
    text!(ax, j+1, (n_row - i + 2),
          text = string(A[i, j]) * (j == n_col ? ";" : ""),
          align = (:center, :center),
          color = :black,
          fontsize = 16)
end
hidedecorations!(ax)  # 軸ラベル、目盛り、グリッドをすべて非表示
hidespines!(ax)       # 軸の枠線を非表示

################################################################
# 各セルに数値を表示
################################################################
ax = Axis(fig[2, 3],
          aspect = DataAspect(),
          title = "with text!(ax, i, j, text = string(A[i, j]))",
          )
# 各セルに数値を表示
for i in 1:size(A, 1), j in 1:size(A, 2)
    text!(ax, i, j, text = string(A[i, j]),
          align = (:center, :center),
          color = :black,
          fontsize = 16)
end

################################################################
# text
################################################################
ax = Axis(fig[1, 5:6],
          aspect = DataAspect(),
          title = "To show a heatmap as in an array literal in Julia code",
          subtitle = "heatmap!(ax, A') with Axis(..., yreversed = true)",
          )
hidedecorations!(ax)  # 軸ラベル、目盛り、グリッドをすべて非表示
hidespines!(ax)       # 軸の枠線を非表示

################################################################
# heatmap!(ax, A as in Julia code)
################################################################
ax = Axis(fig[2, 5:6],
          aspect = DataAspect(),
          title = "heatmap!(ax, A')",
          xlabel = "with yreversed = true",
          yreversed = true,
          )
hm = heatmap!(ax, A', colormap = :viridis)

fig |> display
safesave(plotsdir(program_name * ".pdf"), fig)
