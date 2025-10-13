using DrWatson
@quickactivate "BayesianStatisticsWithJuliaSuyama"
program_name = "p095-uniform"
using CairoMakie
using LaTeXStrings
using Distributions

################################################################
# NegativeBinomial
################################################################
a = 0
b = 1
d = Uniform(a, b)

X = rand(d, 100)
mean(X, dims=1)
mean(d)

################################################################
# plot init
################################################################
fig = Figure(size = (800, 600))
Label(fig[0, 1],
      "uniform distribution",
      # L"η = %$η,\text{ }x_{\text{init}} = %$x_init",
      fontsize = 16, font = :bold)
ax = Axis(fig[1,1],
          title = "Uniform($a, $b)",
          xlabel = L"x",
          ylabel = "relative frequency",
          # aspect = DataAspect(),
          aspect = 2,
          )
################################################################
# plot
################################################################
xs = 0:0.01:1
hist!(ax, X, normalization = :probability) # normalization doesn't work well because it's continuous
# hist!(ax, X)
lines!(ax, xs, pdf.(d, xs), color = :black, label = "prob dist")
lines!(ax, xs, cdf.(d, xs), color = :black, label = "cdf")
axislegend(ax,
           backgroundcolor=(:white, 0.5),
           framecolor=(:black, 0.5),       # 枠線も透明に
           # framevisible=false,              # または枠線自体を非表示に
)
################################################################
# plot display and save
################################################################
fig |> display
safesave(plotsdir(program_name * ".pdf"), fig)


