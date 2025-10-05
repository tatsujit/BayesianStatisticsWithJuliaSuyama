using DrWatson
@quickactivate "BayesianStatisticsWithJuliaSuyama"
program_name = "p90-multinomial"
using CairoMakie
using LaTeXStrings
using Distributions

################################################################
# multinomial
################################################################
M = 10
d = Multinomial(M, [0.5, 0.3, 0.2])
x = rand(d)
X = rand(d, 100)
mean(X, dims=2) # [5.0; 2.88; 2.12;;]
mean(X, dims=1) # always 3.333... (x*3 = 10)
mean(d)
cov(X, dims=2)
# 3×3 Matrix{Float64}:
#   3.11111  -2.05051   -1.06061
#  -2.05051   2.22788   -0.177374
#  -1.06061  -0.177374   1.23798
cov(d)
# 3×3 Matrix{Float64}:
#   2.5  -1.5  -1.0
#  -1.5   2.1  -0.6
#  -1.0  -0.6   1.6
# d_pdf = [pdf(d, x) for x in 0.0:0.1:1.0]

################################################################
# plot init
################################################################
fig = Figure(size = (800, 600))
Label(fig[0, 1:2],
      "Distributions",
      # L"η = %$η,\text{ }x_{\text{init}} = %$x_init",
      fontsize = 16, font = :bold)

xs = 0:M
vs = [[x1, x2, M - x1 - x2] for x1 in xs, x2 in xs]
probabilities = [pdf(d, v) for v in vs]

ax = Axis(fig[1,1],
          title = "Multinomial($M, [0.5, 0.3, 0.2])",
          xlabel = L"x_1",
          ylabel = L"x_2",
          aspect = DataAspect(),
          )
# 0から始まる座標を指定
x = 0:size(probabilities, 2)-1
y = 0:size(probabilities, 1)-1
hm = heatmap!(ax, x, y, probabilities)
Colorbar(fig[1,2], hm, label = "probability")

fig |> display
