using DrWatson
@quickactivate "BayesianStatisticsWithJuliaSuyama"
program_name = "p83--discrete-distributions"
using CairoMakie
using LaTeXStrings
using Distributions

################################################################
# plot init
################################################################
fig = Figure(size = (900, 600))
Label(fig[0, 1],
      "Distributions",
      # L"η = %$η,\text{ }x_{\text{init}} = %$x_init",
      fontsize = 16, font = :bold)
################################################################
# Bernoulli
################################################################
d = Beta(1.9, 0.1)
d_pdf = [pdf(d, x) for x in 0.0:0.1:1.0]

limits = (nothing, (0, 2))
ax = Axis(fig[1,1],
          title = "probability mass function",
          xlabel = L"x",
          ylabel = "probability",
          limits = limits,
          )
lines!(ax, 0:0.1:1.0, d_pdf)

d = Beta(1.3, 0.7)
d_pdf = [pdf(d, x) for x in 0.0:0.1:1.0]

ax = Axis(fig[1,2],
          title = "probability mass function",
          xlabel = L"x",
          ylabel = "probability",
          limits = limits,
          )
lines!(ax, 0:0.1:1.0, d_pdf)

d = Beta(1.0, 1.0)
d_pdf = [pdf(d, x) for x in 0.0:0.1:1.0]

ax = Axis(fig[1,3],
          title = "probability mass function",
          xlabel = L"x",
          ylabel = "probability",
          limits = limits,
          )
lines!(ax, 0:0.1:1.0, d_pdf)

fig |> display
