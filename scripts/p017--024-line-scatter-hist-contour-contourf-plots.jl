using DrWatson
@quickactivate "BayesianStatisticsWithJuliaSuyama"
program_name = "p17--24-line-scatter-hist-contour-contourf-plots.pdf"
using CairoMakie
using LaTeXStrings

################################################################
# plot only y-values
################################################################
y1 = [1, 7, 11, 13, 15, 16]
y2 = [15, 3, 13, 2, 7, 1]

fig = Figure(size=(700, 800))
ax = Axis(fig[1, 1],
          xlabel = "index",
          ylabel = L"y",
          title = "just y values"
          )
lines!(ax, y1, label = "y1")
lines!(ax, y2, label = "y2")

################################################################
# parametric curves (cycloid)
################################################################
ax = Axis(fig[2, 1],
          xlabel = L"x",
          ylabel = L"y",
          title = "cycloid, parametric"
          )
r = 1.0
fx(θ) = r*(θ-sin(θ))
fy(θ) = r*(1-cos(θ))
θs = range(-3π, 3π, length = 100)
lines!(ax, fx.(θs), fy.(θs))

################################################################
# scatter and histogram plots
################################################################
ax = Axis(fig[1, 2],
          xlabel = L"x",
          ylabel = L"y",
          title = "scatter plot"
          )
xs = randn(100)
ys = rand(100)
scatter!(ax, xs, ys, alpha = 0.3)

ax = Axis(fig[2, 2],
          xlabel = L"x",
          ylabel = "frequency",
          title = "hist with default bins"
          )
hist!(ax, xs)

ax = Axis(fig[3, 2],
          xlabel = L"x",
          ylabel = "frequency",
          title = "hist with bins=50"
          )
hist!(ax, xs, bins=50)

################################################################
# contour map
################################################################
fz(x, y) = exp(-(2x^2 + y^2 + x*y))
xs = range(-1, 1, length=100)
ys = range(-2, 2, length=100)

ax = Axis(fig[4:5, 1],
          xlabel = L"x",
          ylabel = L"y",
          title = "contour - 0.1:0.2:0.9"
          )
contour!(ax, xs, ys, fz,
         levels = 0.1:0.2:0.9,
         linewidth = 1,
         labels = true,
         )

ax = Axis(fig[4:5, 2],
          xlabel = L"x",
          ylabel = L"y",
          title = "contour - exp.(range(log(0.01), log(1), length=5))"
          )
contour!(ax, xs, ys, fz,
         levels = exp.(range(log(0.01), log(1), length=5)),
         linewidth = 1,
         labels = true,
         )

ax = Axis(fig[6:7, 1],
          xlabel = L"x",
          ylabel = L"y",
          title = "contour"
          )
contour!(ax, xs, ys, fz,
         levels = 5,
         linewidth = 1,
         labels = true,
         )
ax = Axis(fig[6:7, 2],
          xlabel = L"x",
          ylabel = L"y",
          title = "contourf"
          )
cf = contourf!(ax, xs, ys, fz,
               levels = 5,
               colormap = :viridis)

Colorbar(fig[6:7, 3], cf, label = L"f(x, y)")

title

fig |> display
save(plotsdir(program_name * ".pdf"), fig)
